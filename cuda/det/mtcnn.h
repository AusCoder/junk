#ifndef _MTCNN_H_
#define _MTCNN_H_

#include "trtNet.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

struct MtcnnPnetBuffers {
  float *resizedImage;
  float *prob;
  float *reg;
};

class Mtcnn {
public:
  Mtcnn(cudaStream_t *stream);
  ~Mtcnn();

  Mtcnn(const Mtcnn &) = delete;
  Mtcnn &operator=(const Mtcnn &) = delete;
  Mtcnn(Mtcnn &&) = delete;
  Mtcnn &operator=(Mtcnn &&) = delete;

  void predict(cv::Mat image, cudaStream_t *stream);

private:
  void stageOne(cv::Mat image, cudaStream_t *stream);

  std::vector<float> computeScales(int height, int width);

  std::map<std::pair<int, int>, TrtNet> pnets;
  // TrtNet onet;
  // TrtNet rnet;

  float *dImage;          // originalImage
  float *dImageResizeBox; // resize original image box
  std::map<std::pair<int, int>, MtcnnPnetBuffers> pnetBuffers;

  int requiredImageHeight = 720;
  int requiredImageWidth = 1280;
  int requiredImageDepth = 3;
  int dImageResizeBoxSize = 4;

  float factor = 0.709f;
  int minSize = 40;
};

#endif
