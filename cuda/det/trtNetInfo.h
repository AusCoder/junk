#ifndef _TRT_NET_INFO_H
#define _TRT_NET_INFO_H

#include "NvInfer.h"

#include <string>
#include <vector>

enum class TensorInputOrder { None, NHWC };

struct TensorInfo {
  TensorInfo(std::string n, nvinfer1::Dims3 s, TensorInputOrder i);
  TensorInfo(std::string n, nvinfer1::Dims3 s);

  std::string name;
  nvinfer1::Dims3 shape;
  TensorInputOrder inputOrder;

  int volume() const;
  std::string render() const;
};

struct TrtNetInfo {
  std::vector<TensorInfo> inputTensorInfos;
  std::vector<TensorInfo> outputTensorInfos;

  static TrtNetInfo readTrtNetInfo(const std::string &netInfoPath);

  std::string render() const;
};

#endif