#include "mtcnn.h"
#include "trtNet.h"

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

void Mtcnn::predict(cv::Mat image) {
  TrtNet pnet;
  pnet.start();
  image.convertTo(image, CV_32FC3);
  Mtcnn::stageOne(image);
}

void Mtcnn::stageOne(cv::Mat image) {
  int imageWidth = image.cols;
  int imageHeight = image.rows;

  std::cout << "Image size: " << imageHeight << " x " << imageWidth
            << std::endl;

  std::vector<float> scales = computeScales(imageWidth, imageHeight);

  cv::cuda::GpuMat gpuImage;
  cv::cuda::GpuMat scaledGpuImage;
  gpuImage.upload(image);

  for (auto &scale : scales) {
    int widthScaled = imageWidth * scale;
    int heightScaled = imageHeight * scale;
    // std::cout << heightScaled << ", " << widthScaled << std::endl;

    cv::Size scaledSize{widthScaled, heightScaled};
    cv::cuda::resize(gpuImage, scaledGpuImage, scaledSize, 0, 0,
                     cv::INTER_LINEAR);

    std::cout << "Cuda image scaled continuous? "
              << scaledGpuImage.isContinuous() << std::endl;
    std::cout << "Cuda image scaled dims: " << scaledGpuImage.rows << " x "
              << scaledGpuImage.cols << std::endl;

    // TODO: get the tensorrt engine going
  }
}

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
