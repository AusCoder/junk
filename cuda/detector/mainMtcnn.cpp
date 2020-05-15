#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "mtcnn.h"

#define IMAGE_PATH "/code/junk/tests/data/elephant.jpg"

int main(int argc, char **argv) {
  std::string imagePath = IMAGE_PATH;
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

  Mtcnn mtcnn;
  mtcnn.predict(image);
  return 0;
}
