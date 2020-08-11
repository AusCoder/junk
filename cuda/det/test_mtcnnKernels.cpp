#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "cnpy.h"
#include "commonCuda.h"
#include "kernelTestHarness.hpp"
#include "mtcnnKernels.h"

#include <algorithm>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#define TOLERANCE 0.0001

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
runGenerateBoxes(const std::vector<float> &prob, const std::vector<float> &reg,
                 int probWidth, int probHeight, float threshold, float scale) {
  int maxOutputBoxes = 200;
  std::vector<float> outProb(maxOutputBoxes);
  std::vector<float> outReg(4 * maxOutputBoxes);
  std::vector<float> outBbox(4 * maxOutputBoxes);

  // std::vector<int> outIndices(200);
  // size_t outIndicesSize = outIndices.size();
  // std::fill(outIndices.begin(), outIndices.end(), -1);
  // std::vector<float> outBboxes(4 * outIndicesSize);
  // size_t outBboxesSize = outBboxes.size();

  assert(probWidth * probHeight * 2 == prob.size());

  KernelTestHarness harness;
  harness.addInput(prob);
  harness.addInput(reg);
  harness.addOutput(outProb);
  harness.addOutput(outReg);
  harness.addOutput(outBbox);

  generateBoxesWithoutSoftmax(harness.getInput<float>(0), probWidth, probHeight,
                              harness.getInput<float>(1), probWidth, probHeight,
                              harness.getOutput<float>(0),
                              harness.getOutput<float>(1),
                              harness.getOutput<float>(2), maxOutputBoxes,
                              threshold, scale, &harness.getStream());

  harness.copyOutput(0, outProb);
  harness.copyOutput(1, outReg);
  harness.copyOutput(2, outBbox);

  return {outProb, outReg, outBbox};
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
}