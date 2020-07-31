#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "cnpy.h"
#include "commonCuda.h"
#include "mtcnnKernels.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#define TOLERANCE 0.0001

std::pair<std::vector<int>, std::vector<float>>
runGenerateBoxes(const std::vector<float> &prob, const std::vector<float> &reg,
                 int probWidth, int probHeight, float threshold, float scale) {
  std::vector<int> outIndices(200);
  size_t outIndicesSize = outIndices.size();
  std::fill(outIndices.begin(), outIndices.end(), -1);
  std::vector<float> outBboxes(4 * outIndicesSize);
  size_t outBboxesSize = outBboxes.size();

  size_t probSize = probWidth * probHeight * 2;
  assert(probSize == prob.size());
  size_t regSize = prob.size();

  float *dProb;
  float *dReg;
  int *dOutIndices;
  float *dOutBboxes;

  // TODO: Create a test case object where these allocations happen in cstors,
  // dstors etc
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  CUDACHECK(
      cudaMalloc(reinterpret_cast<void **>(&dProb), sizeof(float) * probSize));
  CUDACHECK(
      cudaMalloc(reinterpret_cast<void **>(&dReg), sizeof(float) * regSize));
  CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&dOutIndices),
                       sizeof(int) * outIndicesSize));
  CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&dOutBboxes),
                       sizeof(float) * outBboxesSize));

  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(dProb), static_cast<const void *>(prob.data()),
      sizeof(float) * probSize, cudaMemcpyHostToDevice, stream));
  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(dReg), static_cast<const void *>(reg.data()),
      sizeof(float) * regSize, cudaMemcpyHostToDevice, stream));
  CUDACHECK(cudaMemcpyAsync(static_cast<void *>(dOutIndices),
                            static_cast<const void *>(outIndices.data()),
                            sizeof(float) * outIndicesSize,
                            cudaMemcpyHostToDevice, stream));

  generateBoxesWithoutSoftmax(dProb, probWidth, probHeight, dOutIndices,
                              dOutBboxes, outIndicesSize, threshold, scale,
                              &stream);

  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(outIndices.data()), static_cast<void *>(dOutIndices),
      sizeof(float) * outIndicesSize, cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(outBboxes.data()), static_cast<void *>(dOutBboxes),
      sizeof(float) * outBboxesSize, cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaFree(static_cast<void *>(dProb)));
  CUDACHECK(cudaFree(static_cast<void *>(dReg)));
  CUDACHECK(cudaStreamDestroy(stream));

  return {outIndices, outBboxes};
}

// NB: expected outputs can be generated with
//  python main.py run-keras --debug-input-output-dir data/test-input-output
TEST_CASE("mtcnnKernels - generateBoxes") {
  cnpy::NpyArray probArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_prob_1-103-187-2.npy");
  cnpy::NpyArray regArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_reg_1-103-187-4.npy");
  cnpy::NpyArray expectedOutputBoxesArray = cnpy::npy_load(
      "data/test-input-output/generate-boxes_0_output-boxes_190-4.npy");

  std::vector<double> expectedOutBoxes_d =
      expectedOutputBoxesArray.as_vec<double>();
  // Range cstor
  std::vector<float> expectedOutBoxes{expectedOutBoxes_d.begin(),
                                      expectedOutBoxes_d.end()};
  std::vector<size_t> expectedOutBoxesShape = expectedOutputBoxesArray.shape;

  int probWidth = 187;
  int probHeight = 103;
  REQUIRE(probArray.num_bytes() == 1 * 103 * 187 * 2 * sizeof(float));

  float threshold = 0.9;
  float scale = 0.3;
  auto output =
      runGenerateBoxes(probArray.as_vec<float>(), regArray.as_vec<float>(),
                       probWidth, probHeight, threshold, scale);
  auto &outIndices = output.first;
  auto &outBboxes = output.second;

  size_t indexCount = 0;
  std::for_each(outIndices.begin(), outIndices.end(),
                [&indexCount](int idx) -> void {
                  if (idx > -1) {
                    indexCount++;
                  }
                });
  REQUIRE(indexCount == expectedOutBoxesShape.at(0));

  for (size_t i = 0; i < expectedOutBoxes.size(); i++) {
    REQUIRE(abs(expectedOutBoxes.at(i) - outBboxes.at(i)) < TOLERANCE);
  }
}