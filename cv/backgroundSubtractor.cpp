/*
  Background subtractor example.
  Performs background subtraction on a video and shows
  the foreground image diff.
*/

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#define FRAME_WIDTH 1280.0
#define FRAME_HEIGHT 720.0

int main() {
  std::string videoFilepath{"input.ts"};
  auto cap = cv::VideoCapture(videoFilepath);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video capture\n";
    return 1;
  }

  auto backgroundSubtractor = cv::createBackgroundSubtractorMOG2();

  cv::Mat frame, foregroundMask;
  while (true) {
    cap.read(frame);
    if (frame.empty()) {
      break;
    }
    cv::resize(frame, frame, cv::Size(), FRAME_WIDTH / frame.cols,
               FRAME_HEIGHT / frame.rows);

    backgroundSubtractor->apply(frame, foregroundMask);
    // XXX: Do a sum and threshold to tell if something is happening
    // in the scene.

    cv::imshow("Frame", frame);
    cv::imshow("Foreground mask", foregroundMask);
    int k = cv::waitKey(1);
    if ((k & 0xff) == 'q') {
      break;
    }
  }
}
