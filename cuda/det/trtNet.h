/*
Provides an interface for running a TensorRT model.
The goal is to provide a simple interface that avoids
the need for all the TensorRT specific stuff with Engines
and Contexts etc.
*/

#ifndef _TRT_NET_H
#define _TRT_NET_H

#include "NvInfer.h"
#include "NvUffParser.h"
#include "logger.h"

#include <map>
#include <utility>

struct TrtNetInfo {
  TrtNetInfo(nvinfer1::Dims3 in, nvinfer1::Dims3 outProb,
             nvinfer1::Dims3 outReg);
  TrtNetInfo(nvinfer1::Dims3 in, nvinfer1::Dims3 outProb,
             nvinfer1::Dims3 outReg, nvinfer1::Dims3 outLand);

  nvinfer1::Dims3 inputShape;
  nvinfer1::Dims3 outputProbShape;
  nvinfer1::Dims3 outputRegShape;
  nvinfer1::Dims3 outputLandmarksShape;
};

class TrtNet {
public:
  static std::map<std::pair<int, int>, TrtNetInfo> TRT_NET_INFO;

  TrtNet();
  ~TrtNet();

  TrtNet(const TrtNet &) = delete;
  TrtNet &operator=(const TrtNet &) = delete;
  TrtNet(TrtNet &&) = delete;
  TrtNet &operator=(TrtNet &&) = delete;

  void start();

  void predict(float *image, int imageSize, float *outputProb,
               int outputProbSize, float *outputReg, int outputRegSize);

  nvinfer1::Dims3 getInputShape();
  nvinfer1::Dims3 getOutputProbShape();
  nvinfer1::Dims3 getOutputRegShape();

private:
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;

  std::string inputName;
  std::string outputProbName;
  std::string outputRegName;

  nvinfer1::Dims3 inputShape;
  nvinfer1::Dims3 outputProbShape;
  nvinfer1::Dims3 outputRegShape;
};

#endif
