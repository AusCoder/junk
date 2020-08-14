#include <cub/cub.cuh>

#include <algorithm>
#include <iostream>
#include <vector>

template <typename Key, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void simpleSort(Key *inputs, int inputsSize, Key *outputs,
                    int outputsSize) {
  typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoad;
  typedef cub::BlockStore<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockStore;
  typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD>
      BlockRadixSort;

  __shared__ union TempStorage {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockRadixSort::TempStorage sort;
  } temp_storage;

  Key thread_data[ITEMS_PER_THREAD];
  BlockLoad(temp_storage.load).Load(inputs, thread_data, inputsSize, 1000000);

  __syncthreads();

  BlockRadixSort(temp_storage.sort).Sort(thread_data);

  __syncthreads();
  BlockStore(temp_storage.store).Store(outputs, thread_data, outputsSize);
}

template <typename Key, typename Value, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void simpleSortValues(Key *keys, int keysSize, Value *values,
                          int valuesSize, Value *outputs, int outputsSize) {
  // assert(keysSize == valuesSize);

  typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoadKeys;
  typedef cub::BlockLoad<Value, BLOCK_THREADS, ITEMS_PER_THREAD>
      BlockLoadValues;
  typedef cub::BlockStore<Value, BLOCK_THREADS, ITEMS_PER_THREAD> BlockStore;
  typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, Value>
      BlockRadixSort;

  __shared__ union TempStorage {
    typename BlockLoadKeys::TempStorage loadKeys;
    typename BlockLoadValues::TempStorage loadValues;
    typename BlockStore::TempStorage store;
    typename BlockRadixSort::TempStorage sort;
  } temp_storage;

  Key thread_keys[ITEMS_PER_THREAD];
  Value thread_values[ITEMS_PER_THREAD];
  BlockLoadKeys(temp_storage.loadKeys).Load(keys, thread_keys, keysSize, 1000000);
  __syncthreads();
  BlockLoadValues(temp_storage.loadValues).Load(values, thread_values, valuesSize);
  __syncthreads();

  BlockRadixSort(temp_storage.sort).Sort(thread_keys, thread_values);

  __syncthreads();
  BlockStore(temp_storage.store).Store(outputs, thread_values, outputsSize);
}

int main() {
  std::vector<float> hInputs(10);
  std::vector<int> hInputIndices(hInputs.size());
  std::vector<int> hOutputs(hInputs.size());

  std::generate(hInputs.begin(), hInputs.end(), [n = -1]() mutable {
    ++n;
    return n / 10.0;
  });
  std::random_shuffle(hInputs.begin(), hInputs.end());
  std::generate(hInputIndices.begin(), hInputIndices.end(),
                [n = 0]() mutable { return n++; });

  std::cout << "Inputs:\n";
  std::for_each(hInputs.begin(), hInputs.end(),
                [](auto &x) { std::cout << x << ", "; });
  std::cout << "\n";

  float *dInputs = nullptr;
  int *dInputIndices = nullptr;
  int *dOutputs = nullptr;

  CubDebugExit(cudaMalloc(reinterpret_cast<void **>(&dInputs),
                          sizeof(float) * hInputs.size()));
  CubDebugExit(cudaMalloc(reinterpret_cast<void **>(&dInputIndices),
                          sizeof(int) * hInputIndices.size()));
  CubDebugExit(cudaMalloc(reinterpret_cast<void **>(&dOutputs),
                          sizeof(int) * hOutputs.size()));

  CubDebugExit(cudaMemcpy(dInputs, hInputs.data(),
                          sizeof(float) * hInputs.size(),
                          cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(dInputIndices, hInputIndices.data(),
                          sizeof(int) * hInputIndices.size(),
                          cudaMemcpyHostToDevice));

  simpleSortValues<float, int, 128, 4><<<1, 128>>>(
      dInputs, hInputs.size(), dInputIndices, hInputIndices.size(), dOutputs, hOutputs.size());

  CubDebugExit(cudaMemcpy(hOutputs.data(), dOutputs,
                          sizeof(int) * hOutputs.size(),
                          cudaMemcpyDeviceToHost));

  CubDebugExit(cudaFree(dInputs));
  CubDebugExit(cudaFree(dInputIndices));
  CubDebugExit(cudaFree(dOutputs));

  std::cout << "Outputs:\n";
  std::for_each(hOutputs.begin(), hOutputs.end(),
                [](auto &x) { std::cout << x << ", "; });
  std::cout << "\n";
}