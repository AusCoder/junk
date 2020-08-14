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

#include "commonCuda.hpp"
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

static cv::Mat debugGetDeviceImage(DevicePtr<float> image, int imageWidth,
                                   int imageHeight, int depth,
                                   cudaStream_t *stream) {
  cv::Mat mat;
  mat.create(imageHeight, imageWidth, CV_32FC3);

  assert(mat.isContinuous());
  assert(mat.total() * mat.elemSize() ==
         imageHeight * imageWidth * depth * sizeof(float));

  denormalizePixels(image.get(), imageHeight * imageWidth * depth, stream);
  CopyElementsAsync(mat.ptr<float>(), image, imageHeight * imageWidth * depth,
                    *stream);
  CUDACHECK(cudaStreamSynchronize(*stream));

  mat.convertTo(mat, CV_8UC3);
  cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  return mat;
}

static void debugSaveDeviceImage(DevicePtr<float> image, int imageWidth,
                                 int imageHeight, int depth,
                                 const std::string &outPath,
                                 cudaStream_t *stream) {
  auto mat = debugGetDeviceImage(image, imageWidth, imageHeight, depth, stream);
  cv::imwrite(outPath, mat);
}

static void debugSaveDeviceImage(DevicePtr<float> image, int imageWidth,
                                 int imageHeight, int depth,
                                 DevicePtr<float> boxes,
                                 const std::string &outPath,
                                 cudaStream_t *stream) {
  auto mat = debugGetDeviceImage(image, imageWidth, imageHeight, depth, stream);
  auto hBoxes = boxes.asVec(*stream);
  CUDACHECK(cudaStreamSynchronize(*stream));
  for (size_t i = 0; i < hBoxes.size() / 4; i++) {
    cv::Point p1{static_cast<int>(hBoxes.at(4 * i + 0)),
                 static_cast<int>(hBoxes.at(4 * i + 1))};
    cv::Point p2{static_cast<int>(hBoxes.at(4 * i + 2)),
                 static_cast<int>(hBoxes.at(4 * i + 3))};
    cv::rectangle(mat, p1, p2, {0, 255, 0});
  }
  cv::imwrite(outPath, mat);
}

static const TrtNetInfo &
maxSizePnetInfo(const std::map<std::pair<int, int>, TrtNet> &pnets) {
  auto m = std::max_element(pnets.begin(), pnets.end(), [](auto &n1, auto &n2) {
    return n1.first.first < n2.first.first;
  });
  return m->second.getTrtNetInfo();
}

PnetMemory::PnetMemory(int batchSize, int maxBoxesToGenerate,
                       const TrtNetInfo &netInfo)
    : resizedImage{DeviceMemory<float>::AllocateElements(
          batchSize * netInfo.inputTensorInfos[0].volume())},
      prob{DeviceMemory<float>::AllocateElements(
          batchSize * netInfo.outputTensorInfos.at(0).volume())},
      reg{DeviceMemory<float>::AllocateElements(
          batchSize * netInfo.outputTensorInfos.at(1).volume())},
      generateBoxesOutputProb{
          DeviceMemory<float>::AllocateElements(maxBoxesToGenerate)},
      generateBoxesOutputReg{
          DeviceMemory<float>::AllocateElements(4 * maxBoxesToGenerate)},
      generateBoxesOutputBoxes{
          DeviceMemory<float>::AllocateElements(4 * maxBoxesToGenerate)},
      generateBoxesOutputBoxesCount{DeviceMemory<int>::AllocateElements(1)},
      nmsSortIndices{DeviceMemory<int>::AllocateElements(maxBoxesToGenerate)},
      nmsOutputProb{DeviceMemory<float>::AllocateElements(maxBoxesToGenerate)},
      outputReg{DeviceMemory<float>::AllocateElements(4 * maxBoxesToGenerate)},
      outputBoxes{
          DeviceMemory<float>::AllocateElements(4 * maxBoxesToGenerate)},
      outputBoxesCount{DeviceMemory<int>::AllocateElements(1)} {}

Mtcnn::Mtcnn(cudaStream_t &stream)
    : pnets{createPnets()}, dImage{DeviceMemory<float>::AllocateElements(
                                requiredImageWidth * requiredImageHeight *
                                requiredImageDepth)},
      dImageResizeBox{
          DeviceMemory<float>::AllocateElements(dImageResizeBoxSize)},
      pnetMemory{1, maxBoxesToGenerate, maxSizePnetInfo(pnets)} {
  // std::map<std::pair<int, int>, TrtNetInfo> netInfos;
  // This is mapping of a dict and inserting into a new dict
  // std::transform(pnets.begin(), pnets.end(),
  //                std::inserter(netInfos, netInfos.begin()),
  //                [](auto &net) -> std::pair<std::pair<int, int>, TrtNetInfo>
  //                {
  //                  return {net.first, net.second.getTrtNetInfo()};
  //                });

  // std::transform(
  //     netInfos.begin(), netInfos.end(),
  //     std::inserter(pnetMemory, pnetMemory.begin()),
  //     [this](auto &netInfo) -> std::pair<std::pair<int, int>, PnetMemory> {
  //       return {netInfo.first,
  //               PnetMemory(1, maxBoxesToGenerate, netInfo.second)};
  //     });

  std::for_each(pnets.begin(), pnets.end(),
                [](auto &net) -> void { net.second.start(); });

  CopyAllElementsAsync(static_cast<DevicePtr<float>>(dImageResizeBox),
                       {0.0, 0.0, 1.0, 1.0}, stream);
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

  CopyElementsAsync(dImage, image.ptr<float>(),
                    imageHeight * imageWidth * depth, *stream);
  normalizePixels(dImage, *stream);
  // Agrees with python up to here

  SetElementAsync(pnetMemory.outputBoxesCount, 0, *stream);

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
    // auto &buffers = pnetMemory.at(resizedHeightWidth);

    assert(pnetMemory.resizedImage.size() ==
           static_cast<std::size_t>(resizedHeightWidth.first *
                                    resizedHeightWidth.second * depth));
    cropResizeHWC(dImage, imageWidth, imageHeight, depth, dImageResizeBox,
                  dImageResizeBoxSize, resizedHeightWidth.second,
                  resizedHeightWidth.first, pnetMemory.resizedImage,
                  resizedHeightWidth.first * resizedHeightWidth.second * depth,
                  *stream);
    // Different resizing algorithm means that we differ from python at this
    // point

    // debugSaveDeviceImage(pnetMemory.resizedImage, resizedHeightWidth.second,
    //                      resizedHeightWidth.first, depth,
    //                      "deviceResizedImage.jpg", stream);

    pnet.predict({pnetMemory.resizedImage.get()},
                 {pnetMemory.prob.get(), pnetMemory.reg.get()}, 1, stream);

    auto &outputTensorInfos = pnet.getTrtNetInfo().outputTensorInfos;
    int probWidth = outputTensorInfos.at(0).getWidth();
    int probHeight = outputTensorInfos.at(0).getHeight();
    int regWidth = outputTensorInfos.at(1).getWidth();
    int regHeight = outputTensorInfos.at(1).getHeight();

    generateBoxesWithoutSoftmax(pnetMemory.prob.get(), probWidth, probHeight,
                                pnetMemory.reg.get(), regWidth, regHeight,
                                pnetMemory.generateBoxesOutputProb.get(),
                                pnetMemory.generateBoxesOutputReg.get(),
                                pnetMemory.generateBoxesOutputBoxes.get(),
                                pnetMemory.generateBoxesOutputBoxesCount.get(),
                                maxBoxesToGenerate, threshold1, scale, stream);

    const float iouThreshold = 0.5;
    nms(pnetMemory.generateBoxesOutputProb, pnetMemory.generateBoxesOutputReg,
        pnetMemory.generateBoxesOutputBoxes,
        pnetMemory.generateBoxesOutputBoxesCount, pnetMemory.nmsSortIndices,
        pnetMemory.nmsOutputProb, pnetMemory.outputReg, pnetMemory.outputBoxes,
        pnetMemory.outputBoxesCount, iouThreshold, *stream);

    // debugPrintVals(pnetMemory.outputBoxes.get(), 20, 0, stream);

    // I can probably roll this into the nms kernel
    regressAndSquareBoxes(pnetMemory.outputBoxes, pnetMemory.outputReg,
                          pnetMemory.outputBoxesCount, true, *stream);

    debugSaveDeviceImage(dImage, imageWidth, imageHeight, depth,
                         pnetMemory.outputBoxes, "deviceResizedImage.jpg",
                         stream);

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
