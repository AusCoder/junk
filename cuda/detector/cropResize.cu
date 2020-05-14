/*
Kernel to crop and resize boxes from an image
*/
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common.h"

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
      printf("Unexpected batchIdx: %d\n", batchIdx);
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

    const float topLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              leftXIndex]));
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

vector<float> runCropResize(const vector<float> &image, int imageWidth,
                            int imageHeight, int depth,
                            const vector<float> &boxes, int boxesSize,
                            int cropWidth, int cropHeight) {
  int croppedBoxesSize = boxesSize * cropWidth * cropHeight * depth;
  vector<float> croppedBoxes(croppedBoxesSize);

  float *dImage;
  float *dBoxes;
  float *dCroppedBoxes;

  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * image.size()));
  CUDACHECK(cudaMalloc((void **)&dBoxes, sizeof(float) * boxes.size()));
  CUDACHECK(
      cudaMalloc((void **)&dCroppedBoxes, sizeof(float) * croppedBoxes.size()));

  CUDACHECK(cudaMemcpy((void *)dImage, (void *)image.data(),
                       sizeof(float) * image.size(), cudaMemcpyHostToDevice));

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

  return croppedBoxes;
}

int main(int argc, char **argv) {
  // int imageWidth = 2;
  // int imageHeight = 2;
  // int depth = 3;
  // vector<float> image{1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
  //                     3.0, 3.0, 3.0, 4.0, 4.0, 4.0};

  std::string imagePath =
      "/home/seb/code/ii/ml-source/data/sample_images/execs.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  // image.convertTo(image, cv::CV_32FC3);

  cv::Vec3f point = image.at<cv::Vec3f>(0, 0);
  cout << point[0] << endl;

  // vector<float> boxes = {0.3, 0.3, 0.7, 0.7};
  // int boxesSize = 1;
  // int cropHeight = 9;
  // int cropWidth = 9;

  // auto croppedBoxes = runCropResize(image, imageWidth, imageHeight, depth,
  //                                   boxes, boxesSize, cropHeight, cropWidth);

  // for (auto &val : croppedBoxes) {
  //   cout << val << ", ";
  // }
  // cout << endl;

  return 0;
}
