#ifndef _TRT_NET_INFO_H
#define _TRT_NET_INFO_H

#include "NvInfer.h"

#include <string>
#include <vector>

enum class TensorInputOrder { None, NHWC };

struct TensorInfo {
  TensorInfo(std::string n, std::vector<int> s, TensorInputOrder i);
  TensorInfo(std::string n, std::vector<int> s);

  std::string name;
  // nvinfer1::Dims3 shape;
  std::vector<int> shape;
  TensorInputOrder inputOrder;

  int getHeight() const;
  int getWidth() const;
  int volume() const;
  std::string render() const;
  nvinfer1::Dims3 getInputDims() const;
};

struct TrtNetInfo {
  std::vector<TensorInfo> inputTensorInfos;
  std::vector<TensorInfo> outputTensorInfos;

  static TrtNetInfo readTrtNetInfo(const std::string &netInfoPath);

  std::string render() const;
};

#endif