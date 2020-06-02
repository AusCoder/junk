#define CATCH_CONFIG_MAIN

#include "trtNet.h"

#include "catch.hpp"

#include <vector>

TEST_CASE("TrtNet - zeros") {
  int imageHeight = 216;
  int imageWidth = 384;
  int channels = 3;

  TrtNet net;

  auto netInfo = net.TRT_NET_INFO.at({imageWidth, imageHeight});
  net.start();
  std::vector<float> image(dimsSize(netInfo.inputShape));
  std::vector<float> outputProb(dimsSize(netInfo.outputProbShape));
  std::vector<float> outputReg(dimsSize(netInfo.outputRegShape));
  std::fill(image.begin(), image.end(), 0.0f);

  net.predict(image.data(), imageHeight, imageWidth, channels,
              outputProb.data(), dimsSize(netInfo.outputProbShape),
              outputReg.data(), dimsSize(netInfo.outputRegShape));

  bool foundGtZero = false;
  for (auto &x : outProb) {
    if (x > 0) {
      foundGtZero = true;
      break;
    }
  }
  REQUIRE(foundGtZero);
}

TEST_CASE("TrtNet - pnet - 216x384") {
  int imageHeight = 216;
  int imageWidth = 384;
  int channels = 3;

  TrtNet net;

  auto netInfo = net.TRT_NET_INFO.at({imageWidth, imageHeight});
  net.start();
  std::vector<float> image(dimsSize(net.getInputShape()));
  std::vector<float> outputProb(dimsSize(net.getOutputProbShape()));
  std::vector<float> outputReg(dimsSize(net.getOutputRegShape()));
  std::fill(image.begin(), image.end(), 0.0f);

  net.predict(image.data(), dimsSize(net.getInputShape()), outputProb.data(),
              dimsSize(net.getOutputProbShape()), outputReg.data(),
              dimsSize(net.getOutputRegShape()));

  bool foundGtZero = false;
  for (auto &x : outProb) {
    if (x > 0) {
      foundGtZero = true;
      break;
    }
  }
  REQUIRE(foundGtZero);
}
