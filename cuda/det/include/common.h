#ifndef _COMMON_H_
#define _COMMON_H_

#include <sstream>
#include <string>

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << endl;             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

struct Prob {
  float x;
  float y;

  Prob() = default;
  Prob(float x_, float y_) : x{x_}, y{y_} {}

  std::string toString() {
    std::stringstream ss;
    ss << "Prob(" << x << ", " << y << ")";
    return ss.str();
  }
};

#endif
