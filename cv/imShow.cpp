#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char **argv) {
  std::string imagePath = "/code/junk/tests/data/elephant.jpg";

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

  cv::imshow("Window", image);

  while (true) {
    int k = cv::waitKey(0);
    if ((k & 0xff) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}
