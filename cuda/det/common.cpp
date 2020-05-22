#include "common.h"

#include <iostream>

void channelSwap(const float *image, int imageWidth, int imageHeight, int depth,
                 ChannelSwapType type, float *out) {
  int size = imageHeight * imageWidth * depth;
  if (type == ChannelSwapType::CHWtoHWC) {
    for (int idx = 0; idx < size; idx++) {
      int tmpIdx = idx;
      int x = tmpIdx % imageWidth;
      tmpIdx /= imageWidth;
      int y = tmpIdx % imageHeight;
      tmpIdx /= imageHeight;
      int chan = tmpIdx % depth;

      int outIdx = (y * imageWidth + x) * depth + chan;
      out[outIdx] = image[idx];
    }
  } else if (type == ChannelSwapType::HWCtoCHW) {
    for (int idx = 0; idx < size; idx++) {
      int tmpIdx = idx;
      int chan = tmpIdx % depth;
      tmpIdx /= depth;
      int x = tmpIdx % imageWidth;
      tmpIdx /= imageWidth;
      int y = tmpIdx % imageHeight;

      int outIdx = (chan * imageHeight + y) * imageWidth + x;
      out[outIdx] = image[idx];
    }
  } else {
    throw std::invalid_argument("Unknown ChannelSwapType");
  }
}
