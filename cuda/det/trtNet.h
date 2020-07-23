/*
Provides an interface for running a TensorRT model.
The goal is to provide a simple interface that avoids
the need for all the TensorRT specific stuff with Engines
and Contexts etc.
*/

#ifndef _TRT_NET_H
#define _TRT_NET_H

#include "NvInfer.h"
// #include "NvUffParser.h"
#include "cuda_runtime.h"
#include "logger.h"
#include "trtNetInfo.h"

#include <map>
#include <utility>
#include <vector>

// enum class InputOrder { None, NHWC };

// struct TensorInfo {
//   TensorInfo(std::string n, nvinfer1::Dims3 s, InputOrder i);
//   TensorInfo(std::string n, nvinfer1::Dims3 s);
//   // Are these really required?
//   // TensorInfo(const TensorInfo &tensorInfo);

//   std::string name;
//   nvinfer1::Dims3 shape;
//   InputOrder inputOrder;
// };

// struct TrtNetInfo {
//   std::vector<TensorInfo> inputTensorInfos;
//   std::vector<TensorInfo> outputTensorInfos;
// };

class TrtNet {
public:
  // static std::map<std::pair<int, int>, TrtNetInfo> TRT_NET_INFO;

  TrtNet(const std::string &p, const TrtNetInfo &i);
  ~TrtNet();
  TrtNet(TrtNet &&net);

  TrtNet(const TrtNet &) = delete;
  TrtNet &operator=(const TrtNet &) = delete;
  TrtNet &operator=(TrtNet &&) = delete;

  void start();

  // void predict(const std::vector<float *> inputs);
  void predictFromHost(const std::vector<float *> &inputs,
                       const std::vector<float *> &outputs,
                       cudaStream_t *stream);

  // nvinfer1::Dims3 getInputShape();
  // nvinfer1::Dims3 getOutputProbShape();
  // nvinfer1::Dims3 getOutputRegShape();

  const TrtNetInfo &getTrtNetInfo();
  const TensorInfo &getInputTensorInfo(int i);
  const TensorInfo &getOutputTensorInfo(int i);

  // static TrtNetInfo createPnetInfo();

private:
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;

  std::string modelPath;
  TrtNetInfo trtNetInfo;

  // std::string inputName;
  // std::string outputProbName;
  // std::string outputRegName;

  // nvinfer1::Dims3 inputShape;
  // nvinfer1::Dims3 outputProbShape;
  // nvinfer1::Dims3 outputRegShape;
};

#endif
