#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#define WINDOW_NAME "Live"
#define MODEL_PATH                                                             \
  "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

class FaceDetector {
public:
  FaceDetector() : faceClassifier{MODEL_PATH} {}

  std::vector<cv::Rect> detectFaces(cv::Mat bgrFrame) {
    cv::Mat greyFrame;
    cv::cvtColor(bgrFrame, greyFrame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(greyFrame, greyFrame);

    std::vector<cv::Rect> faces;
    faceClassifier.detectMultiScale(greyFrame, faces);
    return faces;
  };

private:
  cv::CascadeClassifier faceClassifier;
};

int main(int argc, char **argv) {
  std::string videoFilepath{"/data/cortex-extras-data/staging/videos/"
                            "ml-error_3_02122037-9ec6-49c5-bff7-35b00ebf60ef/"
                            "data/video/2020-09-30/1601429081287000000.ts"};

  auto cap = cv::VideoCapture(videoFilepath);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video capture\n";
    return 1;
  }

  auto faceDetector = FaceDetector();

  cv::Mat frame;
  while (true) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "Failed to read frame\n";
      break;
    }

    cv::imshow(WINDOW_NAME, frame);
    int k = cv::waitKey(25);
    if ((k & 0xff) == 'q') {
      break;
    }
  }
}
