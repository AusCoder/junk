#ifndef _COMMON_CUDA_H
#define _COMMON_CUDA_H

#include <exception>
#include <iostream>

#include "cuda_runtime.h"
#include "NvInferRuntimeCommon.h"

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      std::cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << std::endl;             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int dimsSize(nvinfer1::Dims dims);

#endif
