#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "mtcnn.h"

#define IMAGE_PATH "/code/junk/tests/data/execs.jpg"

int main(int argc, char **argv) {
  std::string imagePath = IMAGE_PATH;
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::resize(image, image, {1280, 720});

  Mtcnn mtcnn;
  mtcnn.predict(image);
  return 0;
}
