#include "deviceMemory.hpp"
#include "mtcnnKernels.h"
#include "streamManager.hpp"
#include <algorithm>
#include <vector>

int main() {
  std::vector<float> values(20);
  std::fill(values.begin(), values.end(), 3.0f);

  StreamManager streamManager;

  auto devMemory = DeviceMemory<float>::AllocateElements(values.size());
  DevicePtr<float> devValues{devMemory};
  CopyAllElementsAsync(devValues, values, streamManager.stream());

  scalarMult(devValues, values.size(), 7.0f, streamManager.stream());
  auto outputValues = devValues.asVec(streamManager.stream());

  streamManager.synchronize();

  std::for_each(outputValues.begin(), outputValues.end(),
                [](auto &x) { std::cout << x << ", "; });
  std::cout << "\n";
}