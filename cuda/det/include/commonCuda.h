#ifndef _COMMON_CUDA_H
#define _COMMON_CUDA_H

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << endl;             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#endif
