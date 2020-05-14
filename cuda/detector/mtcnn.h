#ifndef _MTCNN_H_
#define _MTCNN_H_

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

class Mtcnn {
public:
  Mtcnn() = default;
  Mtcnn(const Mtcnn &) = delete;
  Mtcnn &operator=(const Mtcnn &) = delete;
  Mtcnn(Mtcnn &&) = delete;
  Mtcnn &operator=(Mtcnn &&) = delete;

  void predict(cv::Mat image);

private:
  void stageOne(cv::Mat image);

  std::vector<float> computeScales(int height, int width);

  float factor = 0.709f;
  int minSize = 40;
};

#endif
