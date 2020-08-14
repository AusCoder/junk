#ifndef _MTCNN_H_
#define _MTCNN_H_

#include "deviceMemory.hpp"
#include "trtNet.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

struct PnetMemory {
  PnetMemory(int batchSize, int maxBoxesToGenerate, const TrtNetInfo &netInfo);

  PnetMemory(const PnetMemory &) = delete;
  PnetMemory &operator=(const PnetMemory &) = delete;

  DeviceMemory<float> resizedImage;
  DeviceMemory<float> prob;
  DeviceMemory<float> reg;

  DeviceMemory<float> generateBoxesOutputProb;
  DeviceMemory<float> generateBoxesOutputReg;
  DeviceMemory<float> generateBoxesOutputBoxes;
  DeviceMemory<int> generateBoxesOutputBoxesCount;

  DeviceMemory<int> nmsSortIndices;
  DeviceMemory<float> nmsOutputProb;

  // TODO: rename these outputPorb, outputReg, outputBoxes?
  // Actually, we just care about the output boxes really
  DeviceMemory<float> outputReg;
  DeviceMemory<float> outputBoxes;
  DeviceMemory<int> outputBoxesCount;
};

class Mtcnn {
public:
  Mtcnn(cudaStream_t &stream);

  Mtcnn(const Mtcnn &) = delete;
  Mtcnn &operator=(const Mtcnn &) = delete;
  Mtcnn(Mtcnn &&) = delete;
  Mtcnn &operator=(Mtcnn &&) = delete;

  void predict(cv::Mat image, cudaStream_t *stream);

private:
  void stageOne(cv::Mat image, cudaStream_t *stream);

  std::vector<float> computeScales(int height, int width);

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

  std::map<std::pair<int, int>, TrtNet> pnets;
  // TrtNet onet;
  // TrtNet rnet;

  // float *dImage; // originalImage
  // float *dImageResizeBox; // resize original image box
  DeviceMemory<float> dImage;
  DeviceMemory<float> dImageResizeBox;
  PnetMemory pnetMemory;
  // std::map<std::pair<int, int>, MtcnnPnetMemory> pnetMemory;
};

#endif
