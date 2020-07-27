#ifndef _COMMON_CUDA_H
#define _COMMON_CUDA_H

#include "NvInferRuntimeCommon.h"
#include "cuda_runtime.h"

#include <iostream>

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      std::cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << std::endl;   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define DEBUG_PRINT(msg, value)                                                \
  do {                                                                         \
    std::cout << msg << value << "\n"                                          \
  } while (0)

#define DEBUG_PRINT_VEC(msg, value)                                            \
  do {                                                                         \
    std::cout << msg;                                                          \
    for (auto &x : value) {                                                    \
      std::cout << x << ", ";                                                  \
    }                                                                          \
    std::cout << "\n";                                                         \
  } while (0)

int dimsSize(nvinfer1::Dims dims);

#endif
