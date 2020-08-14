#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "cnpy.h"
#include "deviceMemory.hpp"
#include "mtcnnKernels.h"
#include "streamManager.hpp"

#include <algorithm>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#define TOLERANCE 0.0001

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, int>
runGenerateBoxes(const std::vector<float> &prob, const std::vector<float> &reg,
                 int probWidth, int probHeight, float threshold, float scale) {
  int maxOutputBoxes = 200;
  std::vector<float> outProb(maxOutputBoxes);
  std::vector<float> outReg(4 * maxOutputBoxes);
  std::vector<float> outBbox(4 * maxOutputBoxes);
  int outBoxesCount;

  StreamManager streamManager;

  assert(static_cast<std::size_t>(probWidth * probHeight * 2) == prob.size());

  auto dProb = DeviceMemory<float>::AllocateElements(prob.size());
  CopyAllElementsAsync(dProb, prob, streamManager.stream());
  auto dReg = DeviceMemory<float>::AllocateElements(reg.size());
  CopyAllElementsAsync(dReg, reg, streamManager.stream());

  auto dOutProb = DeviceMemory<float>::AllocateElements(outProb.size());
  auto dOutReg = DeviceMemory<float>::AllocateElements(outReg.size());
  auto dOutBoxes = DeviceMemory<float>::AllocateElements(outBbox.size());
  auto dOutBoxesCount = DeviceMemory<int>::AllocateElements(1);

  generateBoxesWithoutSoftmax(
      dProb.get(), probWidth, probHeight, dReg.get(), probWidth, probHeight,
      dOutProb.get(), dOutReg.get(), dOutBoxes.get(), dOutBoxesCount.get(),
      maxOutputBoxes, threshold, scale, &streamManager.stream());

  CopyAllElementsAsync(outProb, dOutProb, streamManager.stream());
  CopyAllElementsAsync(outReg, dOutReg, streamManager.stream());
  CopyAllElementsAsync(outBbox, dOutBoxes, streamManager.stream());
  CopyAllElementsAsync(&outBoxesCount, dOutBoxesCount, streamManager.stream());

  return {outProb, outReg, outBbox, outBoxesCount};
}

TEST_CASE("mtcnnKernels - generateBoxes") {
  cnpy::NpyArray probArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_prob_1-103-187-2.npy");
  cnpy::NpyArray regArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_reg_1-103-187-4.npy");
  cnpy::NpyArray expectedOutputProbArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_output-prob_190.npy");
  cnpy::NpyArray expectedOutputRegArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_output-reg_190-4.npy");
  cnpy::NpyArray expectedOutputBoxesArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_output-boxes_190-4.npy");

  std::vector<float> expectedOutProb = expectedOutputProbArray.as_vec<float>();
  std::vector<float> expectedOutReg = expectedOutputRegArray.as_vec<float>();
  std::vector<float> expectedOutBoxes =
      expectedOutputBoxesArray.as_vec<float>();

  std::vector<size_t> expectedOutBoxesShape = expectedOutputBoxesArray.shape;

  int probWidth = 187;
  int probHeight = 103;
  REQUIRE(probArray.num_bytes() == 1 * 103 * 187 * 2 * sizeof(float));

  float threshold = 0.9;
  float scale = 0.3;
  auto output =
      runGenerateBoxes(probArray.as_vec<float>(), regArray.as_vec<float>(),
                       probWidth, probHeight, threshold, scale);
  auto &outProb = std::get<0>(output);
  auto &outReg = std::get<1>(output);
  auto &outBboxes = std::get<2>(output);

  for (size_t i = 0; i < std::min(expectedOutProb.size(), outProb.size());
       i++) {
    REQUIRE(abs(expectedOutProb.at(i) - outProb.at(i)) < TOLERANCE);
  }
  for (size_t i = 0; i < std::min(expectedOutReg.size(), outReg.size()); i++) {
    REQUIRE(abs(expectedOutReg.at(i) - outReg.at(i)) < TOLERANCE);
  }
  for (size_t i = 0; i < std::min(expectedOutBoxes.size(), outBboxes.size());
       i++) {
    REQUIRE(abs(expectedOutBoxes.at(i) - outBboxes.at(i)) < TOLERANCE);
  }
  REQUIRE(static_cast<size_t>(std::get<3>(output)) ==
          expectedOutBoxesShape.at(0));
}

TEST_CASE("mtcnnKernels - regressAndSquareBoxes - regress only") {
  cnpy::NpyArray inputBoxesArray = cnpy::npy_load(
      "data/test-input-output/regress-and-square_input-box_102-4.npy");
  cnpy::NpyArray inputRegArray = cnpy::npy_load(
      "data/test-input-output/regress-and-square_input-reg_102-4.npy");
  cnpy::NpyArray expectedBoxesWithReqArray = cnpy::npy_load(
      "data/test-input-output/regress-and-square_output-reg-boxes_102-4.npy");
  cnpy::NpyArray expectedBoxesWithRegAndSqArray =
      cnpy::npy_load("data/test-input-output/"
                     "regress-and-square_output-reg-and-sq-boxes_102-4.npy");

  StreamManager streamManager;
  auto inputBoxes = inputBoxesArray.as_vec<float>();
  std::vector<int> inputBoxesShape{inputBoxesArray.shape.begin(),
                                   inputBoxesArray.shape.end()};
  auto inputReg = inputRegArray.as_vec<float>();
  auto dInputBoxes = DeviceMemory<float>::AllocateElements(inputBoxes.size());
  auto dInputBoxesCount = DeviceMemory<int>::AllocateElements(1);
  auto dInputReg = DeviceMemory<float>::AllocateElements(inputReg.size());
  CopyAllElementsAsync(dInputBoxes, inputBoxes, streamManager.stream());
  CopyElementsAsync(dInputBoxesCount, inputBoxesShape.data(), 1,
                    streamManager.stream());
  CopyAllElementsAsync(dInputReg, inputReg, streamManager.stream());

  regressAndSquareBoxes(dInputBoxes, dInputReg, dInputBoxesCount, false,
                        streamManager.stream());

  auto expectedBoxes = expectedBoxesWithReqArray.as_vec<float>();
  auto outputBoxes = dInputBoxes.asVec(streamManager.stream());
  streamManager.synchronize();
  REQUIRE(outputBoxes.size() == expectedBoxes.size());
  for (std::size_t i = 0; i < outputBoxes.size(); i++) {
    REQUIRE(abs(outputBoxes.at(i) - expectedBoxes.at(i)) < TOLERANCE);
  }
}

TEST_CASE("mtcnnKernels - regressAndSquareBoxes - regress and square") {
  cnpy::NpyArray inputBoxesArray = cnpy::npy_load(
      "data/test-input-output/regress-and-square_input-box_102-4.npy");
  cnpy::NpyArray inputRegArray = cnpy::npy_load(
      "data/test-input-output/regress-and-square_input-reg_102-4.npy");
  cnpy::NpyArray expectedBoxesWithRegAndSqArray =
      cnpy::npy_load("data/test-input-output/"
                     "regress-and-square_output-reg-and-sq-boxes_102-4.npy");

  StreamManager streamManager;
  auto inputBoxes = inputBoxesArray.as_vec<float>();
  std::vector<int> inputBoxesShape{inputBoxesArray.shape.begin(),
                                   inputBoxesArray.shape.end()};
  auto inputReg = inputRegArray.as_vec<float>();
  auto dInputBoxes = DeviceMemory<float>::AllocateElements(inputBoxes.size());
  auto dInputBoxesCount = DeviceMemory<int>::AllocateElements(1);
  auto dInputReg = DeviceMemory<float>::AllocateElements(inputReg.size());
  CopyAllElementsAsync(dInputBoxes, inputBoxes, streamManager.stream());
  CopyElementsAsync(dInputBoxesCount, inputBoxesShape.data(), 1,
                    streamManager.stream());
  CopyAllElementsAsync(dInputReg, inputReg, streamManager.stream());

  regressAndSquareBoxes(dInputBoxes, dInputReg, dInputBoxesCount, true,
                        streamManager.stream());

  auto expectedBoxes = expectedBoxesWithRegAndSqArray.as_vec<float>();
  auto outputBoxes = dInputBoxes.asVec(streamManager.stream());
  streamManager.synchronize();
  REQUIRE(expectedBoxes.size() == outputBoxes.size());
  for (std::size_t i = 0; i < outputBoxes.size(); i++) {
    REQUIRE(abs(outputBoxes.at(i) - expectedBoxes.at(i)) < TOLERANCE);
  }
}