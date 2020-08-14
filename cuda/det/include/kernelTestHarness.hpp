#ifndef _KERNEL_TEST_HARNESS_HPP
#define _KERNEL_TEST_HARNESS_HPP

#include "deviceMemory.hpp"
#include "streamManager.hpp"
#include <cassert>
#include <vector>

class KernelTestHarness {
public:
  // KernelTestHarness() { CUDACHECK(cudaStreamCreate(&stream)); }

  // ~KernelTestHarness() {
  //   for (auto &ptr : deviceInputs) {
  //     CUDACHECK(cudaFree(ptr));
  //   }
  //   for (auto &ptr : deviceOutputs) {
  //     CUDACHECK(cudaFree(ptr));
  //   }
  //   CUDACHECK(cudaStreamDestroy(stream));
  // }

  template <typename InputType>
  void addInput(const std::vector<InputType> &input) {
    void *ptr;
    CUDACHECK(cudaMalloc(&ptr, sizeof(InputType) * input.size()));
    CUDACHECK(cudaMemcpyAsync(ptr, static_cast<const void *>(input.data()),
                              sizeof(InputType) * input.size(),
                              cudaMemcpyHostToDevice, stream));
    deviceInputs.push_back(ptr);
  }

  template <typename OutputType> void addOutput(size_t outputSize) {
    void *ptr;
    size_t size = sizeof(OutputType) * outputSize;
    CUDACHECK(cudaMalloc(&ptr, size));
    deviceOutputs.push_back(ptr);
    deviceOutputSizes.push_back(size);
  }

  template <typename OutputType>
  void addOutput(const std::vector<OutputType> &output) {
    addOutput<OutputType>(output.size());
    assert(deviceOutputSizes.at(deviceOutputSizes.size() - 1) ==
           sizeof(OutputType) * output.size());
    void *deviceOutput = deviceOutputs.at(deviceOutputs.size() - 1);
    CUDACHECK(
        cudaMemcpyAsync(deviceOutput, static_cast<const void *>(output.data()),
                        deviceOutputSizes.at(deviceOutputSizes.size() - 1),
                        cudaMemcpyHostToDevice, stream));
  }

  template <typename InputType> InputType *getInput(int idx) {
    return static_cast<InputType *>(deviceInputs.at(idx));
  }

  template <typename OutputType> OutputType *getOutput(int idx) {
    return static_cast<OutputType *>(deviceOutputs.at(idx));
  }

  template <typename OutputType>
  void copyOutput(int idx, std::vector<OutputType> &output) {
    assert(sizeof(OutputType) * output.size() >= deviceOutputSizes.at(idx));
    CUDACHECK(cudaMemcpyAsync(static_cast<void *>(output.data()),
                              deviceOutputs.at(idx), deviceOutputSizes.at(idx),
                              cudaMemcpyDeviceToHost, stream));
  }

  int getNumOutputs() { return deviceOutputs.size(); }

  cudaStream_t &getStream() { return stream; }

private:
  // cudaStream_t stream;
  // std::vector<void *> deviceInputs;
  // std::vector<void *> deviceOutputs;
  // std::vector<int> deviceOutputSizes;

  StreamManager streamManager;
  // std::vector<
};

#endif
