#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mtcnn.h"

#define IMAGE_PATH "/home/seb/Pictures/mountains.jpg"

int main(int argc, char **argv) {
  std::string imagePath = IMAGE_PATH;
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

  Mtcnn mtcnn;
  mtcnn.predict(image);
  return 0;
}
