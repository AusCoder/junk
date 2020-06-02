#define CATCH_CONFIG_MAIN

#include "commonCuda.h"
#include "trtNet.h"

#include "catch.hpp"

#include <vector>

TEST_CASE("TrtNet - zeros") {
  int imageHeight = 216;
  int imageWidth = 384;
  int channels = 3;

  TrtNet net;

  net.start();
  std::vector<float> image(dimsSize(net.getInputShape()));
  std::vector<float> outputProb(dimsSize(net.getOutputProbShape()));
  std::vector<float> outputReg(dimsSize(net.getOutputRegShape()));
  std::fill(image.begin(), image.end(), 0.0f);

  net.predict(image.data(), dimsSize(net.getInputShape()), outputProb.data(),
              dimsSize(net.getOutputProbShape()), outputReg.data(),
              dimsSize(net.getOutputRegShape()));

  bool foundGtZero = false;
  for (auto &x : outputProb) {
    if (x > 0) {
      foundGtZero = true;
      break;
    }
  }
  REQUIRE(foundGtZero);
}

// TEST_CASE("TrtNet - pnet - 216x384") {
//   int imageHeight = 216;
//   int imageWidth = 384;
//   int channels = 3;

//   TrtNet net;

//   net.start();
//   std::vector<float> image(dimsSize(net.getInputShape()));
//   std::vector<float> outputProb(dimsSize(net.getOutputProbShape()));
//   std::vector<float> outputReg(dimsSize(net.getOutputRegShape()));
//   std::fill(image.begin(), image.end(), 0.0f);

//   net.predict(image.data(), dimsSize(net.getInputShape()), outputProb.data(),
//               dimsSize(net.getOutputProbShape()), outputReg.data(),
//               dimsSize(net.getOutputRegShape()));

//   bool foundGtZero = false;
//   for (auto &x : outputProb) {
//     if (x > 0) {
//       foundGtZero = true;
//       break;
//     }
//   }
//   REQUIRE(foundGtZero);
// }
