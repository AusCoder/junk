#include "commonCuda.hpp"

#include <stdexcept>

int dimsSize(nvinfer1::Dims dims) {
  if (dims.nbDims == -1) {
    throw std::invalid_argument("dims.nbDims is -1");
  }
  int size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }
  return size;
}
