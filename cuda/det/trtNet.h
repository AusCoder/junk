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

class TrtNet {
public:
  TrtNet(int maxBatchSize_, const std::string &p, const TrtNetInfo &i);
  ~TrtNet();
  TrtNet(TrtNet &&net);
  TrtNet &operator=(TrtNet &&net);

  TrtNet(const TrtNet &) = delete;
  TrtNet &operator=(const TrtNet &) = delete;

  void start();

  // TODO: Handle maxBatchSize and batchSize
  void predict(const std::vector<float *> &inputs,
               const std::vector<float *> &outputs, int batchSize,
               cudaStream_t *stream);
  void predictFromHost(const std::vector<float *> &inputs,
                       const std::vector<float *> &outputs, int batchSize,
                       cudaStream_t *stream);

  const TrtNetInfo &getTrtNetInfo() const;
  const TensorInfo &getInputTensorInfo(int i) const;
  const TensorInfo &getOutputTensorInfo(int i) const;

  static TrtNet createFromUffAndInfoFile(int maxBatchSize,
                                         const std::string &uffPath);

private:
  int maxBatchSize;
  bool isStarted = false;
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;

  std::string modelPath;
  TrtNetInfo trtNetInfo;
};

#endif
