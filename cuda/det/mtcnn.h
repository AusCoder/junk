#ifndef _MTCNN_H_
#define _MTCNN_H_

#include "trtNet.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

struct MtcnnPnetBuffers {
  // TODO: add more sizes here
  float *resizedImage;
  float *prob;
  float *reg;
  float *generateBoxesOutputProb;
  size_t generateBoxesOutputProbSize;
  float *generateBoxesOutputReg;
  size_t generateBoxesOutputRegSize;
  float *generateBoxesOutputBoxes;
  size_t generateBoxesOutputBoxesSize;
  int *nmsSortIndices;
  size_t nmsSortIndicesSize;
  float *nmsOutputProb;
  size_t nmsOutputProbSize;
  float *nmsOutputReg;
  size_t nmsOutputRegSize;
  float *nmsOutputBoxes;
  size_t nmsOutputBoxesSize;
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

  int maxBoxesToGenerate = 500;

  float factor = 0.709f;
  int minSize = 40;
  float threshold1 = 0.9;
  float threshold2 = 0.95;
  float threshold3 = 0.95;
};

#endif
