#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "commonCuda.h"
#include "mtcnn.h"

#define IMAGE_PATH "/code/junk/tests/data/execs.jpg"

int main(int argc, char **argv) {
  std::string imagePath = IMAGE_PATH;
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  cv::resize(image, image, {1280, 720});

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  Mtcnn mtcnn{&stream};
  mtcnn.predict(image, &stream);

  CUDACHECK(cudaStreamDestroy(stream));
  return 0;
}
