#ifndef _COMMON_CUDA_H
#define _COMMON_CUDA_H

#include <exception>

#include "NvInferRuntimeCommon.h"

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << endl;             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int dimsSize(nvinfer1::Dims dims);

#endif
