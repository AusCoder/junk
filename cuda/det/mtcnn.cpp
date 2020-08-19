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
  std::vector<std::string> uffPaths{
      "data/uff/pnet_14x25.uff",   "data/uff/pnet_20x35.uff",
      "data/uff/pnet_28x49.uff",   "data/uff/pnet_39x69.uff",
      "data/uff/pnet_55x98.uff",   "data/uff/pnet_77x137.uff",
      "data/uff/pnet_109x194.uff", "data/uff/pnet_154x273.uff",
      "data/uff/pnet_216x384.uff"};
  std::map<std::pair<int, int>, TrtNet> pnets;
  for (auto &uffPath : uffPaths) {
    TrtNet net{TrtNet::createFromUffAndInfoFile(1, uffPath)};
    const auto &tensorInfo = net.getTrtNetInfo().inputTensorInfos.at(0);
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
                                 DevicePtr<int> boxesCount,
                                 const std::string &outPath,
                                 cudaStream_t *stream) {
  auto mat = debugGetDeviceImage(image, imageWidth, imageHeight, depth, stream);
  auto hBoxes = boxes.asVec(*stream);
  auto hBoxesCount = GetElementAsync(boxesCount, *stream);
  CUDACHECK(cudaStreamSynchronize(*stream));
  for (int i = 0; i < hBoxesCount; i++) {
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
          batchSize * netInfo.inputTensorInfos.at(0).volume())},
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

RnetMemory::RnetMemory(int maxBoxesToProcess, const TrtNetInfo &netInfo)
    : croppedImages{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.inputTensorInfos.at(0).volume())},
      prob{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.outputTensorInfos.at(0).volume())},
      reg{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.outputTensorInfos.at(1).volume())},
      maskOutProb{DeviceMemory<float>::AllocateElements(maxBoxesToProcess)},
      maskOutReg{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      maskOutBoxes{
          DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      maskOutBoxesCount{DeviceMemory<int>::AllocateElements(1)},
      nmsSortIndices{DeviceMemory<int>::AllocateElements(maxBoxesToProcess)},
      nmsOutProb{DeviceMemory<float>::AllocateElements(maxBoxesToProcess)},
      outReg{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      outBoxes{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      outBoxesCount{DeviceMemory<int>::AllocateElements(1)} {
  assert(netInfo.inputTensorInfos.at(0).volume() == 24 * 24 * 3);
  assert(prob.size() == maxBoxesToProcess * 2);
  assert(reg.size() == maxBoxesToProcess * 4);
}

OnetMemory::OnetMemory(int maxBoxesToProcess, const TrtNetInfo &netInfo)
    : croppedImages{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.inputTensorInfos.at(0).volume())},
      prob{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.outputTensorInfos.at(0).volume())},
      reg{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.outputTensorInfos.at(1).volume())},
      landmarks{DeviceMemory<float>::AllocateElements(
          maxBoxesToProcess * netInfo.outputTensorInfos.at(2).volume())},
      maskOutProb{DeviceMemory<float>::AllocateElements(maxBoxesToProcess)},
      maskOutReg{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      maskOutBoxes{
          DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      maskOutBoxesCount{DeviceMemory<int>::AllocateElements(1)},
      nmsSortIndices{DeviceMemory<int>::AllocateElements(maxBoxesToProcess)},
      nmsOutProb{DeviceMemory<float>::AllocateElements(maxBoxesToProcess)},
      outReg{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      outBoxes{DeviceMemory<float>::AllocateElements(4 * maxBoxesToProcess)},
      outBoxesCount{DeviceMemory<int>::AllocateElements(1)} {}

// TODO: is the batch size of rnet way too big?
// Should I make it smaller and batch the input?
Mtcnn::Mtcnn(cudaStream_t &stream)
    : pnets{createPnets()}, rnet{TrtNet::createFromUffAndInfoFile(
                                maxBoxesToGenerate, "data/uff/rnet.uff")},
      onet{TrtNet::createFromUffAndInfoFile(maxBoxesToGenerate,
                                            "data/uff/onet.uff")},
      dImage{DeviceMemory<float>::AllocateElements(
          requiredImageWidth * requiredImageHeight * requiredImageDepth)},
      dImageResizeBox{DeviceMemory<float>::AllocateElements(4)},
      dImageResizeBoxCount{DeviceMemory<int>::AllocateElements(1)},
      pnetMemory{1, maxBoxesToGenerate, maxSizePnetInfo(pnets)},
      rnetMemory{maxBoxesToGenerate, rnet.getTrtNetInfo()},
      onetMemory{maxBoxesToGenerate, onet.getTrtNetInfo()} {
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
  rnet.start();
  onet.start();

  CopyAllElementsAsync(dImageResizeBox,
                       {0.0, 0.0, requiredImageWidth, requiredImageHeight},
                       stream);
  SetElementAsync(dImageResizeBoxCount, 1, stream);
}

void Mtcnn::predict(cv::Mat image, cudaStream_t *stream) {
  image.convertTo(image, CV_32FC3);
  stageOne(image, stream);
  stageTwo(*stream);
  stageThree(*stream);
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

    // auto &resizedHeightWidth = pnetWithSize.first;
    auto &resizedHeight = pnetWithSize.first.first;
    auto &resizedWidth = pnetWithSize.first.second;
    auto &pnet = pnetWithSize.second;
    float scale = static_cast<float>(resizedHeight) / imageHeight;
    std::cout << scale << ", " << resizedHeight << ", " << resizedWidth << "\n";
    // auto &buffers = pnetMemory.at(resizedHeightWidth);

    // assert(pnetMemory.resizedImage.size() ==
    //        static_cast<std::size_t>(resizedHeightWidth.first *
    //                                 resizedHeightWidth.second * depth));
    cropResizeHWC(dImage, requiredImageWidth, requiredImageHeight,
                  requiredImageDepth, dImageResizeBox, dImageResizeBoxCount,
                  resizedWidth, resizedHeight, pnetMemory.resizedImage,
                  resizedWidth * resizedHeight * depth, *stream);
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
    // TODO: think about what to do if we overrun the outputBoxes size
  }
  // I can probably roll this into the nms kernel
  regressAndSquareBoxes(pnetMemory.outputBoxes, pnetMemory.outputReg,
                        pnetMemory.outputBoxesCount, true, *stream);

  // debugSaveDeviceImage(dImage, requiredImageWidth, requiredImageHeight,
  //                      requiredImageDepth, pnetMemory.outputBoxes,
  //                      pnetMemory.outputBoxesCount, "deviceImagePnet.jpg",
  //                      stream);

  CUDACHECK(cudaStreamSynchronize(*stream));
}

void Mtcnn::stageTwo(cudaStream_t &stream) {
  auto &inputTensorInfos = rnet.getTrtNetInfo().inputTensorInfos;
  auto cropWidth = inputTensorInfos.at(0).getWidth();
  auto cropHeight = inputTensorInfos.at(0).getHeight();
  assert(cropWidth == 24);
  assert(cropHeight == 24);

  cropResizeHWC(
      dImage, requiredImageWidth, requiredImageHeight, requiredImageDepth,
      pnetMemory.outputBoxes, pnetMemory.outputBoxesCount, cropWidth,
      cropHeight, rnetMemory.croppedImages,
      maxBoxesToGenerate * cropWidth * cropHeight * requiredImageDepth, stream);

  // TODO: here I could either run the entirety of rnetMemory.croppedImages (ie
  // maxBoxesToGenerate images) or could do a copy from
  // pnetMemory.outputBoxesCount and use that as the batch size
  rnet.predict({rnetMemory.croppedImages.get()},
               {rnetMemory.prob.get(), rnetMemory.reg.get()},
               maxBoxesToGenerate, &stream);

  probMask(rnetMemory.prob, rnetMemory.reg, pnetMemory.outputBoxes,
           pnetMemory.outputBoxesCount, rnetMemory.maskOutProb,
           rnetMemory.maskOutReg, rnetMemory.maskOutBoxes,
           rnetMemory.maskOutBoxesCount, maxBoxesToGenerate, threshold2,
           stream);

  float iouThreshold = 0.7;
  nms(rnetMemory.maskOutProb, rnetMemory.maskOutReg, rnetMemory.maskOutBoxes,
      rnetMemory.maskOutBoxesCount, rnetMemory.nmsSortIndices,
      rnetMemory.nmsOutProb, rnetMemory.outReg, rnetMemory.outBoxes,
      rnetMemory.outBoxesCount, iouThreshold, stream);

  regressAndSquareBoxes(rnetMemory.outBoxes, rnetMemory.outReg,
                        rnetMemory.outBoxesCount, true, stream);

  // debugSaveDeviceImage(dImage, requiredImageWidth, requiredImageHeight,
  //                      requiredImageDepth, rnetMemory.outBoxes,
  //                      rnetMemory.outBoxesCount, "deviceImageStageTwo.jpg",
  //                      &stream);
}

void Mtcnn::stageThree(cudaStream_t &stream) {
  auto &inputTensorInfos = onet.getTrtNetInfo().inputTensorInfos;
  auto cropWidth = inputTensorInfos.at(0).getWidth();
  auto cropHeight = inputTensorInfos.at(0).getHeight();
  assert(cropWidth == 48);
  assert(cropHeight == 48);

  cropResizeHWC(
      dImage, requiredImageWidth, requiredImageHeight, requiredImageDepth,
      rnetMemory.outBoxes, rnetMemory.outBoxesCount, cropWidth, cropHeight,
      onetMemory.croppedImages,
      maxBoxesToGenerate * cropWidth * cropHeight * requiredImageDepth, stream);

  onet.predict(
      {onetMemory.croppedImages.get()},
      {onetMemory.prob.get(), onetMemory.reg.get(), onetMemory.landmarks.get()},
      maxBoxesToGenerate, &stream);

  probMask(onetMemory.prob, onetMemory.reg, rnetMemory.outBoxes,
           rnetMemory.outBoxesCount, onetMemory.maskOutProb,
           onetMemory.maskOutReg, onetMemory.maskOutBoxes,
           onetMemory.maskOutBoxesCount, maxBoxesToGenerate, threshold2,
           stream);

  regressAndSquareBoxes(onetMemory.maskOutBoxes, onetMemory.maskOutReg,
                        onetMemory.maskOutBoxesCount, false, stream);
  float iouThreshold = 0.6;
  nms(onetMemory.maskOutProb, onetMemory.maskOutReg, onetMemory.maskOutBoxes,
      onetMemory.maskOutBoxesCount, onetMemory.nmsSortIndices,
      onetMemory.nmsOutProb, onetMemory.outReg, onetMemory.outBoxes,
      onetMemory.outBoxesCount, iouThreshold, stream);

  debugSaveDeviceImage(dImage, requiredImageWidth, requiredImageHeight,
                       requiredImageDepth, onetMemory.outBoxes,
                       onetMemory.outBoxesCount, "deviceImageStageThree.jpg",
                       &stream);
  CUDACHECK(cudaStreamSynchronize(stream));
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
