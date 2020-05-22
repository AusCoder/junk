#ifndef _COMMON_H_
#define _COMMON_H_

#include <sstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

enum class ChannelSwapType { CHWtoHWC, HWCtoCHW };

void channelSwap(const float *image, int imageWidth, int imageHeight, int depth,
                 ChannelSwapType type, float *out);

#endif
