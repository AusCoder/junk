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

struct RnetMemory {
  RnetMemory(int maxBoxesToProcess, const TrtNetInfo &netInfo);

  RnetMemory(const RnetMemory &) = delete;
  RnetMemory &operator=(const RnetMemory &) = delete;

  DeviceMemory<float> croppedImages;

  DeviceMemory<float> prob;
  DeviceMemory<float> reg;

  DeviceMemory<float> maskOutProb;
  DeviceMemory<float> maskOutReg;
  DeviceMemory<float> maskOutBoxes;
  DeviceMemory<int> maskOutBoxesCount;

  DeviceMemory<int> nmsSortIndices;
  DeviceMemory<float> nmsOutProb;
  DeviceMemory<float> outReg;
  DeviceMemory<float> outBoxes;
  DeviceMemory<int> outBoxesCount;
};

struct OnetMemory {
  OnetMemory(int maxBoxesToProcess, const TrtNetInfo &netInfo);

  OnetMemory(const OnetMemory &) = delete;
  OnetMemory &operator=(const OnetMemory &) = delete;

  DeviceMemory<float> croppedImages;

  DeviceMemory<float> prob;
  DeviceMemory<float> reg;
  DeviceMemory<float> landmarks;

  DeviceMemory<float> maskOutProb;
  DeviceMemory<float> maskOutReg;
  DeviceMemory<float> maskOutBoxes;
  DeviceMemory<int> maskOutBoxesCount;

  DeviceMemory<int> nmsSortIndices;
  DeviceMemory<float> nmsOutProb;
  DeviceMemory<float> outReg;
  DeviceMemory<float> outBoxes;
  DeviceMemory<int> outBoxesCount;
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
  void stageTwo(cudaStream_t &stream);
  void stageThree(cudaStream_t &stream);

  std::vector<float> computeScales(int height, int width);

  int requiredImageHeight = 720;
  int requiredImageWidth = 1280;
  int requiredImageDepth = 3;

  int maxBoxesToGenerate = 500;

  float factor = 0.709f;
  int minSize = 40;
  float threshold1 = 0.9;
  float threshold2 = 0.95;
  float threshold3 = 0.95;

  std::map<std::pair<int, int>, TrtNet> pnets;
  TrtNet rnet;
  TrtNet onet;

  DeviceMemory<float> dImage;
  DeviceMemory<float> dImageResizeBox;
  DeviceMemory<int> dImageResizeBoxCount;
  PnetMemory pnetMemory;
  RnetMemory rnetMemory;
  OnetMemory onetMemory;
};

#endif
