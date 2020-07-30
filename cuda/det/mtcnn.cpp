#include "mtcnn.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <iterator>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "commonCuda.h"
#include "mtcnnKernels.h"

static std::map<std::pair<int, int>, TrtNet> createPnets() {
  std::vector<std::string> uffPaths{"data/uff/pnet_216x384.uff"};
  std::map<std::pair<int, int>, TrtNet> pnets;
  for (auto &uffPath : uffPaths) {
    TrtNet net{TrtNet::createFromUffAndInfoFile(uffPath)};
    const auto &tensorInfo = net.getTrtNetInfo().inputTensorInfos[0];
    auto inputHeightWidth =
        std::make_pair(tensorInfo.getHeight(), tensorInfo.getWidth());
    pnets.emplace(std::make_pair(inputHeightWidth, std::move(net)));
  }
  return pnets;
}

// TODO: move to a MtcnnPnetBuffers cstor
static MtcnnPnetBuffers allocatePnetBuffers(int batchSize,
                                            const TrtNetInfo &netInfo) {
  MtcnnPnetBuffers pnetBuffers;
  // TODO: write these buffer sizes somewhere too for checking
  CUDACHECK(cudaMalloc((void **)&pnetBuffers.resizedImage,
                       sizeof(float) * batchSize *
                           netInfo.inputTensorInfos[0].volume()));
  CUDACHECK(cudaMalloc((void **)&pnetBuffers.prob,
                       sizeof(float) * batchSize *
                           netInfo.outputTensorInfos[0].volume()));
  CUDACHECK(cudaMalloc((void **)&pnetBuffers.reg,
                       sizeof(float) * batchSize *
                           netInfo.outputTensorInfos[1].volume()));
  return pnetBuffers;
}

static void debugDenormalizeAndSaveDeviceImage(float *dImage, int imageWidth,
                                               int imageHeight, int depth,
                                               const std::string &outPath,
                                               cudaStream_t *stream) {
  cv::Mat mat;
  mat.create(imageHeight, imageWidth, CV_32FC3);

  assert(mat.isContinuous());
  assert(mat.total() * mat.elemSize() ==
         imageHeight * imageWidth * depth * sizeof(float));

  denormalizePixels(dImage, imageHeight * imageWidth * depth, stream);
  CUDACHECK(cudaMemcpyAsync(static_cast<void *>(mat.ptr<float>()),
                            static_cast<void *>(dImage),
                            sizeof(float) * imageHeight * imageWidth * depth,
                            cudaMemcpyDeviceToHost, *stream));
  CUDACHECK(cudaStreamSynchronize(*stream));

  mat.convertTo(mat, CV_8UC3);
  cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  cv::imwrite(outPath, mat);
}

Mtcnn::Mtcnn(cudaStream_t *stream) : pnets{createPnets()} {
  std::map<std::pair<int, int>, TrtNetInfo> netInfos;
  // This is mapping of a dict and inserting into a new dict
  std::transform(pnets.begin(), pnets.end(),
                 std::inserter(netInfos, netInfos.begin()),
                 [](auto &net) -> std::pair<std::pair<int, int>, TrtNetInfo> {
                   return {net.first, net.second.getTrtNetInfo()};
                 });
  std::transform(
      netInfos.begin(), netInfos.end(),
      std::inserter(pnetBuffers, pnetBuffers.begin()),
      [](auto &netInfo) -> std::pair<std::pair<int, int>, MtcnnPnetBuffers> {
        return {netInfo.first, allocatePnetBuffers(1, netInfo.second)};
      });
  std::for_each(pnets.begin(), pnets.end(),
                [](auto &net) -> void { net.second.start(); });

  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * requiredImageHeight *
                                             requiredImageWidth *
                                             requiredImageDepth));
  CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&dImageResizeBox),
                       sizeof(float) * dImageResizeBoxSize));
  std::vector<float> imageResizeBox{0.0, 0.0, 1.0, 1.0};
  assert(imageResizeBox.size() == dImageResizeBoxSize);
  CUDACHECK(cudaMemcpyAsync(static_cast<void *>(dImageResizeBox),
                            static_cast<void *>(imageResizeBox.data()),
                            sizeof(float) * imageResizeBox.size(),
                            cudaMemcpyHostToDevice, *stream));
}

Mtcnn::~Mtcnn() {
  CUDACHECK(cudaFree(dImage));
  CUDACHECK(cudaFree(dImageResizeBox));
  for (auto &buffers : pnetBuffers) {
    CUDACHECK(cudaFree(buffers.second.resizedImage));
    CUDACHECK(cudaFree(buffers.second.prob));
    CUDACHECK(cudaFree(buffers.second.reg));
  }
}

void Mtcnn::predict(cv::Mat image, cudaStream_t *stream) {
  image.convertTo(image, CV_32FC3);
  Mtcnn::stageOne(image, stream);
}

void Mtcnn::stageOne(cv::Mat image, cudaStream_t *stream) {
  // TODO: add image size check, add format check to make sure it is
  // CV_32FC3
  // Question: if I do mat.convertTo(mat, CV_32FC3) and mat is already
  // CV_32FC3 will it reallocate?
  if (image.type() != CV_32FC3) {
    assert(false);
    image.convertTo(image, CV_32FC3);
  }
  if (image.rows != requiredImageHeight || image.cols != requiredImageWidth) {
    assert(false);
    cv::resize(image, image, {requiredImageWidth, requiredImageHeight});
  }

  int imageWidth = image.cols;
  int imageHeight = image.rows;
  int depth = image.channels();

  std::cout << "Image size: " << imageHeight << " x " << imageWidth
            << std::endl;

  if (!image.isContinuous()) {
    std::cout << "Warning: image is not continuous, will clone image\n";
    image = image.clone();
  }

  CUDACHECK(cudaMemcpyAsync(static_cast<void *>(dImage),
                            static_cast<void *>(image.ptr<float>()),
                            sizeof(float) * imageHeight * imageWidth * depth,
                            cudaMemcpyHostToDevice, *stream));
  normalizePixels(dImage, imageHeight * imageWidth * depth, stream);
  // Agrees with python up to here

  for (auto &pnetWithSize : pnets) {
    // Want to allocate gpu buffers up front as much as possible
    // Buffers
    // - initial image buffer
    // - buffer for resized image (same as pnet input buffer)
    // - pnet output buffers
    // - other buffers for nms indices (as required)
    //    - makes sense to roll lots of these into a mega kernel I think

    auto &resizedHeightWidth = pnetWithSize.first;
    auto &pnet = pnetWithSize.second;
    float scale = static_cast<float>(resizedHeightWidth.first) / imageHeight;
    std::cout << scale << ", " << resizedHeightWidth.first << ", "
              << resizedHeightWidth.second << "\n";
    auto &buffers = pnetBuffers.at(resizedHeightWidth);

    cropResizeHWC(dImage, imageWidth, imageHeight, depth, dImageResizeBox,
                  dImageResizeBoxSize, resizedHeightWidth.second,
                  resizedHeightWidth.first, buffers.resizedImage,
                  resizedHeightWidth.first * resizedHeightWidth.second * depth,
                  stream);
    // Different resizing algorithm means that we differ from python at this
    // point

    // debugDenormalizeAndSaveDeviceImage(
    //     buffers.resizedImage, resizedHeightWidth.second,
    //     resizedHeightWidth.first, depth, "deviceResizedImage.jpg", stream);

    pnet.predict({buffers.resizedImage}, {buffers.prob, buffers.reg}, 1,
                 stream);
    debugPrintVals(buffers.prob, 10, 0, stream);
    debugPrintVals(buffers.reg, 10, 0, stream);

    // Things might be different at this point. Want to verify:
    // - Original image is the same
    // - After normalizing is the same
    // - After resize is the same
  }
  CUDACHECK(cudaStreamSynchronize(*stream));
}

// Run stageOne with opencv GpuMat resizing
//
// void Mtcnn::stageOne(cv::Mat image) {
//   int imageWidth = image.cols;
//   int imageHeight = image.rows;

//   std::cout << "Image size: " << imageHeight << " x " << imageWidth
//             << std::endl;

//   std::vector<float> scales = computeScales(imageWidth, imageHeight);

//   cv::cuda::GpuMat gpuImage;
//   cv::cuda::GpuMat scaledGpuImage;
//   gpuImage.upload(image);

//   for (auto &scale : scales) {
//     int widthScaled = ceil(imageWidth * scale);
//     int heightScaled = ceil(imageHeight * scale);
//     std::cout << heightScaled << ", " << widthScaled << std::endl;

//     cv::Size scaledSize{widthScaled, heightScaled};
//     cv::cuda::resize(gpuImage, scaledGpuImage, scaledSize, 0, 0,
//                      cv::INTER_LINEAR);

//     if (!scaledGpuImage.isContinuous()) {
//       scaledGpuImage = scaledGpuImage.clone();
//     }

//     std::cout << "Cuda image scaled continuous? "
//               << scaledGpuImage.isContinuous() << std::endl;
//     std::cout << "Cuda image scaled dims: " << scaledGpuImage.rows << " x "
//               << scaledGpuImage.cols << std::endl;

//     // TODO: get the tensorrt engine going
//   }
// }

std::vector<float> Mtcnn::computeScales(int width, int height) {
  std::vector<float> scales;
  float scale = 12.0f / minSize;
  float curSide = std::min(width, height) * scale;

  std::cout << scale << " " << minSize << " " << curSide << std::endl;

  while (curSide >= 12.0f) {
    scales.push_back(scale);
    scale *= factor;
    curSide *= factor;
  }
  return scales;
}
