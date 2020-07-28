#include "mtcnn.h"

#include <cmath>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

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

Mtcnn::Mtcnn() : pnets{createPnets()} {
  for (auto &p : pnets) {
    p.second.start();
  }
}

void Mtcnn::predict(cv::Mat image) {
  image.convertTo(image, CV_32FC3);
  Mtcnn::stageOne(image);
}

void Mtcnn::stageOne(cv::Mat image) {
  int imageWidth = image.cols;
  int imageHeight = image.rows;

  std::cout << "Image size: " << imageHeight << " x " << imageWidth
            << std::endl;

  std::vector<float> scales = computeScales(imageWidth, imageHeight);

  for (auto &scale : scales) {
    // Want to allocate gpu buffers up front as much as possible
    // Buffers
    // - initial image buffer
    // - buffer for resized image (same as pnet input buffer)
    // - pnet output buffers
    // - other buffers for nms indices (as required)
    //    - makes sense to roll lots of these into a mega kernel I think

    int widthScaled = ceil(imageWidth * scale);
    int heightScaled = ceil(imageHeight * scale);
    std::cout << heightScaled << ", " << widthScaled << std::endl;

    cv::Size scaledSize{widthScaled, heightScaled};
    cv::cuda::resize(gpuImage, scaledGpuImage, scaledSize, 0, 0,
                     cv::INTER_LINEAR);

    if (!scaledGpuImage.isContinuous()) {
      scaledGpuImage = scaledGpuImage.clone();
    }

    std::cout << "Cuda image scaled continuous? "
              << scaledGpuImage.isContinuous() << std::endl;
    std::cout << "Cuda image scaled dims: " << scaledGpuImage.rows << " x "
              << scaledGpuImage.cols << std::endl;

    // TODO: get the tensorrt engine going
  }
}

// Run stageOne with opencv GpuMat resizing
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
