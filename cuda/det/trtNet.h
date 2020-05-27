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

class TrtNet {
public:
  TrtNet() = default;
  ~TrtNet();

  TrtNet(const TrtNet &) = delete;
  TrtNet &operator=(const TrtNet &) = delete;
  TrtNet(TrtNet &&) = delete;
  TrtNet &operator=(TrtNet &&) = delete;

  void start();
  // void predict();

  void predict(float *image, int height, int width, float *outArr,
               int outArrSize);

private:
  nvinfer1::IBuilder *builder = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;
};

#endif
