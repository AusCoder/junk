#define CATCH_CONFIG_MAIN

#include "trtNet.h"

#include "catch.hpp"

#include <vector>

TEST_CASE("TrtNet - zeros") {
  int imageHeight = 216;
  int imageWidth = 384;

  TrtNet net;

  auto netInfo = net.TRT_NET_INFO.at({imageWidth, imageHeight});
  REQUIRE(netInfo.inputShape.d[0] == imageWidth);
  // net.start();
  // int imageSize = imageHeight * imageWidth * channels;
  // std::vector<float> image(imageSize);
  // std::fill(image.begin(), image.end(), 0.0f);

  // net.predict(image.data(), imageHeight, imageWidth, channels)
}
