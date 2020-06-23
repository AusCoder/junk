#ifndef _MTCNN_H_
#define _MTCNN_H_

#include "trtNet.h"

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class Mtcnn {
public:
  Mtcnn();
  ~Mtcnn() = default;

  Mtcnn(const Mtcnn &) = delete;
  Mtcnn &operator=(const Mtcnn &) = delete;
  Mtcnn(Mtcnn &&) = delete;
  Mtcnn &operator=(Mtcnn &&) = delete;

  void predict(cv::Mat image);

private:
  void stageOne(cv::Mat image);

  std::vector<float> computeScales(int height, int width);

  std::map<std::pair<int, int>, TrtNet> pnets;
  // TrtNet onet;
  // TrtNet rnet;

  float factor = 0.709f;
  int minSize = 40;
};

#endif
