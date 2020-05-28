/*
Kernel to crop and resize boxes from an image
*/
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common.h"
#include "commonCuda.h"

using namespace std;

// blockSize will be {1024, 1, 1}
// gridSize...
__global__ void cropResizeKernel(
    const float *image, int imageWidth, int imageHeight, int depth,
    const float *boxes, int boxesSize, int cropWidth, int cropHeight,
    float *croppedBoxes,
    int croppedBoxesSize // == boxesSize * cropWidth * cropHeight * depth
) {
  // Assume that the depth is 3 for now

  const float extrapolationValue = 0.0f;
  const int batch = 1;

  // Each thread will loop and write to certain Idx in croppedBoxes
  for (int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
       outIdx < croppedBoxesSize; outIdx += blockDim.x * gridDim.x) {
    int idx = outIdx;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int depthIdx = idx % depth;
    const int boxIdx = idx / depth;

    const float y1 = boxes[boxIdx * 4];
    const float x1 = boxes[boxIdx * 4 + 1];
    const float y2 = boxes[boxIdx * 4 + 2];
    const float x2 = boxes[boxIdx * 4 + 3];

    const int batchIdx = boxIdx / boxesSize;
    if (batchIdx < 0 || batchIdx >= batch) {
      continue;
    }

    const float heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const float widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    const float inY = (cropHeight > 1)
                          ? y1 * (imageHeight - 1) + y * heightScale
                          : 0.5 * (y1 + y2) * (imageHeight - 1);
    if (inY < 0 || inY > imageHeight - 1) {
      croppedBoxes[outIdx] = extrapolationValue;
      continue;
    }

    const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
    if (inX < 0 || inX > imageWidth - 1) {
      croppedBoxes[outIdx] = extrapolationValue;
      continue;
    }

    const int topYIndex = floorf(inY);
    const int bottomYIndex = ceilf(inY);
    const float yLerp = inY - topYIndex;
    const int leftXIndex = floorf(inX);
    const int rightXIndex = ceilf(inX);
    const float xLerp = inX - leftXIndex;

    int topLeftIndex = ((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
      imageWidth + leftXIndex;
    const float topLeft(static_cast<float>(image[topLeftIndex]));
    const float topRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              rightXIndex]));
    const float bottomLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              leftXIndex]));
    const float bottomRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              rightXIndex]));

    const float top = topLeft + (topRight - topLeft) * xLerp;
    const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
    croppedBoxes[outIdx] = top + (bottom - top) * yLerp;
  }
}

vector<float> runCropResize(const float *image, int imageWidth,
                            int imageHeight, int depth,
                            const vector<float> &boxes, int boxesSize,
                            int cropWidth, int cropHeight) {
  vector<float> transposedImage(imageHeight * imageWidth * depth);
  channelSwap(image, imageWidth, imageHeight, depth,
              ChannelSwapType::HWCtoCHW, transposedImage.data());

                        cout << "Running crop Resize" << endl
                              << "  imageWidth:  " << imageWidth << endl
                              << "  imageHeight: " << imageHeight << endl
                              << "  depth:       " << depth << endl
                              << "  boxesSize:   " << boxesSize << endl
                              << "  cropWidth:   " << cropWidth << endl
                              << "  cropHeight:  " << cropHeight << endl;
  cout << "transposedImage: " << transposedImage[0] << " " << transposedImage[1]
       << " " << transposedImage[2] << endl;

  int croppedBoxesSize = boxesSize * cropWidth * cropHeight * depth;
  vector<float> croppedBoxes(croppedBoxesSize);

  float *dImage;
  float *dBoxes;
  float *dCroppedBoxes;

  CUDACHECK(
      cudaMalloc((void **)&dImage, sizeof(float) * transposedImage.size()));
  CUDACHECK(cudaMalloc((void **)&dBoxes, sizeof(float) * boxes.size()));
  CUDACHECK(
      cudaMalloc((void **)&dCroppedBoxes, sizeof(float) * croppedBoxes.size()));

  CUDACHECK(cudaMemcpy((void *)dImage, (void *)transposedImage.data(),
                       sizeof(float) * transposedImage.size(),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy((void *)dBoxes, (void *)boxes.data(),
                       sizeof(float) * boxes.size(), cudaMemcpyHostToDevice));

  const int block = 1024;
  const int grid = (croppedBoxesSize + block - 1) / block;

  cropResizeKernel<<<grid, block>>>(dImage, imageWidth, imageHeight, depth,
                                    dBoxes, boxesSize, cropWidth, cropHeight,
                                    dCroppedBoxes, croppedBoxesSize);

  CUDACHECK(cudaMemcpy((void *)croppedBoxes.data(), (void *)dCroppedBoxes,
                       sizeof(float) * croppedBoxes.size(),
                       cudaMemcpyDeviceToHost));

  CUDACHECK(cudaFree((void *)dImage));
  CUDACHECK(cudaFree((void *)dBoxes));
  CUDACHECK(cudaFree((void *)dCroppedBoxes));

  cout << "croppedBoxes: " << croppedBoxes[0] << " " << croppedBoxes[1] << " "
       << croppedBoxes[2] << endl;

  vector<float> transposedCroppedBoxes(croppedBoxes.size());
  for (int i = 0; i < boxesSize; i++) {
    int offset = i * cropHeight * cropWidth * depth;
    float *crop = croppedBoxes.data() + offset;
    float *transposedCrop = transposedCroppedBoxes.data() + offset;
    channelSwap(crop, cropWidth, cropHeight, depth, ChannelSwapType::CHWtoHWC,
                transposedCrop);
  }

  return transposedCroppedBoxes;
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

  vector<float> boxes = {0.1, 0.1, 0.5, 0.5};
  int boxesSize = 1;
  int cropHeight = 250;
  int cropWidth = 250;

  auto croppedBoxesArr = runCropResize(image.ptr<float>(), imageWidth, imageHeight, depth,
                                       boxes, boxesSize, cropHeight, cropWidth);
  cv::Mat croppedImage{cropHeight, cropWidth, CV_32FC3, croppedBoxesArr.data()};
  cv::Vec3f point = croppedImage.at<cv::Vec3f>(0);
  cout << point[0] << " " << point[1] << " " << point[2] << endl;

  croppedImage.convertTo(croppedImage, CV_8UC3);
  cv::imwrite("cropTest.jpg", croppedImage);

  return 0;
}