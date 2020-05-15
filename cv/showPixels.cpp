#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

#define IMAGE_PATH "/code/junk/tests/data/elephant.jpg"
#define NUM_PIXELS 3

int main(int argc, char **argv) {
  string imagePath = IMAGE_PATH;

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);

  // Show pixels using at
  cout << "Pixels using at" << endl;
  for (int i = 0; i < NUM_PIXELS; i++) {
    int y = i % image.cols;
    int x = i / image.cols;
    cv::Point3f pt = image.at<cv::Point3f>(x, y);
    cout << pt.x << ", " << pt.y << ", " << pt.z << endl;
  }

  // Show pixels using ptr
  cout << "Pixels using ptr" << endl;
  assert(image.isContinuous());
  float *pixelPtr = image.ptr<float>();
  for (int i = 0; i < 3 * NUM_PIXELS; i++) {
    cout << pixelPtr[i] << ", ";
  }
  cout << endl;

  return 0;
}
