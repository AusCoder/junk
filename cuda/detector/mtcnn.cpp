#include "mtcnn.h"

#include <iostream>

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
    int widthScaled = imageWidth * scale;
    int heightScaled = imageHeight * scale;
    std::cout << heightScaled << ", " << widthScaled << std::endl;
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
