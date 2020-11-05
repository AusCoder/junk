#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define WINDOW_NAME "Window"

using namespace std;

const int hMax = 180;
const int sMax = 255;
const int vMax = 255;
std::vector<int> lowerBound;
std::vector<int> upperBound;
cv::Mat image;
cv::Mat mask;

static void onTrackbarChange(int value, void *userData) {
  cv::inRange(image, lowerBound, upperBound, mask);
  cv::imshow(WINDOW_NAME, mask);
}

int main(int argc, char **argv) {
  string imagePath{"/home/seb/Pictures/People.jpeg"};

  lowerBound = {60, 70, 70};
  upperBound = {110, 255, 255};

  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::blur(image, image, {10, 10});
  cv::cvtColor(image, image, cv::COLOR_BGR2HSV);

  cv::createTrackbar("Lower h", WINDOW_NAME, &lowerBound[0], hMax,
                     onTrackbarChange);
  cv::createTrackbar("Lower s", WINDOW_NAME, &lowerBound[1], sMax,
                     onTrackbarChange);
  cv::createTrackbar("Lower v", WINDOW_NAME, &lowerBound[2], vMax,
                     onTrackbarChange);
  cv::createTrackbar("Upper h", WINDOW_NAME, &upperBound[0], hMax,
                     onTrackbarChange);
  cv::createTrackbar("Upper s", WINDOW_NAME, &upperBound[1], sMax,
                     onTrackbarChange);
  cv::createTrackbar("Upper v", WINDOW_NAME, &upperBound[2], vMax,
                     onTrackbarChange);
  onTrackbarChange(0, nullptr);

  while (true) {
    int k = cv::waitKey(0);
    if ((k & 0xff) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}
