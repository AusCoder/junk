#define CATCH_CONFIG_MAIN

#include "commonCuda.hpp"
#include "trtNet.h"

#include "catch.hpp"

#include <vector>

TEST_CASE("TrtNet - zeros") {
  TrtNetInfo netInfo{
      TrtNetInfo::readTrtNetInfo("data/uff/pnet_216x384-info.json")};

  TrtNet net{"data/uff/pnet_216x384.uff", netInfo};
  net.start();

  std::vector<float *> inputs;
  std::vector<float *> outputs;

  std::vector<float> image(netInfo.inputTensorInfos[0].volume());
  std::fill(image.begin(), image.end(), 0.0f);
  inputs.push_back(image.data());

  float outputProb[netInfo.outputTensorInfos[0].volume()];
  float outputReg[netInfo.outputTensorInfos[1].volume()];
  outputs.push_back(outputProb);
  outputs.push_back(outputReg);

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  net.predictFromHost(inputs, outputs, 1, &stream);

  // TODO: add numpy lib so we can verify inputs and outputs agree with keras
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
