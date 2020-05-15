#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
  std::string imagePath = "/home/seb/Pictures/mountains.jpg";

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::Mat resizedImage;
  cv::Size size{250, 500};

  cv::resize(image, resizedImage, size, 0, 0, cv::INTER_LINEAR);

  cv::imshow("Window", resizedImage);

  while (true) {
    int k = cv::waitKey(0);
    if ((k & 0xff) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}
