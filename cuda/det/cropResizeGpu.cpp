/*
Kernel to crop and resize boxes from an image
*/
#include <iostream>
#include <vector>

#include "common.h"
#include "commonCuda.h"
#include "cropResizeKernel.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

vector<float> runCropResize(const float *image, int imageWidth, int imageHeight,
                            int depth, const vector<float> &boxes,
                            int boxesSize, int cropWidth, int cropHeight,
                            bool useTranspose) {

  cout << "Running crop Resize" << endl
       << "  imageWidth:  " << imageWidth << endl
       << "  imageHeight: " << imageHeight << endl
       << "  depth:       " << depth << endl
       << "  boxesSize:   " << boxesSize << endl
       << "  cropWidth:   " << cropWidth << endl
       << "  cropHeight:  " << cropHeight << endl;

  size_t imageSize = imageHeight * imageWidth * depth;
  int croppedBoxesSize = boxesSize * cropWidth * cropHeight * depth;
  vector<float> croppedBoxes(croppedBoxesSize);

  float *dImage;
  float *dBoxes;
  float *dCroppedBoxes;

  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * imageSize));
  CUDACHECK(cudaMalloc((void **)&dBoxes, sizeof(float) * boxes.size()));
  CUDACHECK(
      cudaMalloc((void **)&dCroppedBoxes, sizeof(float) * croppedBoxes.size()));

  if (useTranspose) {
    vector<float> transposedImage(imageSize);
    channelSwap(image, imageWidth, imageHeight, depth,
                ChannelSwapType::HWCtoCHW, transposedImage.data());
    CUDACHECK(cudaMemcpy((void *)dImage, (void *)transposedImage.data(),
                         sizeof(float) * transposedImage.size(),
                         cudaMemcpyHostToDevice));
  } else {
    CUDACHECK(cudaMemcpy((void *)dImage, (void *)image,
                         sizeof(float) * imageSize, cudaMemcpyHostToDevice));
  }

  CUDACHECK(cudaMemcpy((void *)dBoxes, (void *)boxes.data(),
                       sizeof(float) * boxes.size(), cudaMemcpyHostToDevice));

  if (useTranspose) {
    cropResizeCHW(dImage, imageWidth, imageHeight, depth, dBoxes, boxesSize,
                  cropWidth, cropHeight, dCroppedBoxes, croppedBoxesSize);
  } else {
    cropResizeHWC(dImage, imageWidth, imageHeight, depth, dBoxes, boxesSize,
                  cropWidth, cropHeight, dCroppedBoxes, croppedBoxesSize);
  }

  CUDACHECK(cudaMemcpy((void *)croppedBoxes.data(), (void *)dCroppedBoxes,
                       sizeof(float) * croppedBoxes.size(),
                       cudaMemcpyDeviceToHost));

  CUDACHECK(cudaFree((void *)dImage));
  CUDACHECK(cudaFree((void *)dBoxes));
  CUDACHECK(cudaFree((void *)dCroppedBoxes));

  cout << "croppedBoxes: " << croppedBoxes[0] << " " << croppedBoxes[1] << " "
       << croppedBoxes[2] << endl;

  if (useTranspose) {
    vector<float> transposedCroppedBoxes(croppedBoxes.size());
    for (int i = 0; i < boxesSize; i++) {
      int offset = i * cropHeight * cropWidth * depth;
      float *crop = croppedBoxes.data() + offset;
      float *transposedCrop = transposedCroppedBoxes.data() + offset;
      channelSwap(crop, cropWidth, cropHeight, depth, ChannelSwapType::CHWtoHWC,
                  transposedCrop);
    }
    return transposedCroppedBoxes;
  } else {
    return croppedBoxes;
  }
}

int main(int argc, char **argv) {
  std::string imagePath = "../../tests/data/execs.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);

  int imageWidth = image.cols;
  int imageHeight = image.rows;
  int depth = image.channels();

  if (!image.isContinuous()) {
    image = image.clone();
  }

  bool useTranspose = false;

  vector<float> boxes = {0.0, 0.0, 1.0, 1.0};
  int boxesSize = 1;
  int cropHeight = 720;
  int cropWidth = 1280;

  auto croppedBoxesArr =
      runCropResize(image.ptr<float>(), imageWidth, imageHeight, depth, boxes,
                    boxesSize, cropWidth, cropHeight, useTranspose);
  cv::Mat croppedImage{cropHeight, cropWidth, CV_32FC3, croppedBoxesArr.data()};
  cv::Vec3f point = croppedImage.at<cv::Vec3f>(0);
  cout << point[0] << " " << point[1] << " " << point[2] << endl;

  croppedImage.convertTo(croppedImage, CV_8UC3);
  cv::imwrite("cropTest.jpg", croppedImage);

  return 0;
}
