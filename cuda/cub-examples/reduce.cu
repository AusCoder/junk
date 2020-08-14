#include <cub/cub.cuh>

#include <algorithm>
#include <iostream>
#include <vector>

// BLOCK_THREADS === number of threads per block
// ITEMS_PER_THREAD === ... items per thread
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void simpleSum(int *inputs, int *outputs) {
  typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoad;
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;

  // __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ union TempStorage {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
  } temp_storage;

  int thread_data[ITEMS_PER_THREAD];

  BlockLoad(temp_storage.load).Load(inputs, thread_data);

  // Sync because we use shared memory between Load and Sum
  __syncthreads();

  // The total sum ends up in threadIdx.0's agg
  int agg = BlockReduce(temp_storage.reduce).Sum(thread_data);
  if (agg > 0) {
    printf("Aggregate is %d\n", agg);
  }
}

int main() {
  std::vector<int> hInputs(10);
  std::generate(hInputs.begin(), hInputs.end(), [n = -1]() mutable {
    ++n;
    return n;
  });
  std::for_each(hInputs.begin(), hInputs.end(),
                [](auto &x) { std::cout << x << ","; });
  std::cout << "\n";

  int *dInputs = nullptr;
  int *dOutputs = nullptr;

  CubDebugExit(cudaMalloc(reinterpret_cast<void **>(&dInputs),
                          sizeof(int) * hInputs.size()));
  CubDebugExit(cudaMalloc(reinterpret_cast<void **>(&dOutputs),
                          sizeof(int) * hInputs.size()));

  CubDebugExit(cudaMemcpy(dInputs, hInputs.data(), sizeof(int) * hInputs.size(),
                          cudaMemcpyHostToDevice));

  simpleSum<128, 4><<<1, 128>>>(dInputs, dOutputs);

  CubDebugExit(cudaFree(static_cast<void *>(dInputs)));
  CubDebugExit(cudaFree(static_cast<void *>(dOutputs)));
  std::cout << "success\n";
}