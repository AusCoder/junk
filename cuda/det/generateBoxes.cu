/*
Program to attempt mtcnn box generation in cuda.
*/
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cnpy.h"
#include "common.h"

using namespace std;

/*
  Here every thread is looping the whole prob array, I don't
  know of a way around this from a fundamental level.

  I can think of us splitting the prob grid up a lot then looping in
  each block.

  The issue with doing it this way is that we can't hit that much of the
  prob array. Here if we run this kernel with a block size of {256, 1, 1}
  then we will hit just the first 256 prob entries, it might be much bigger
  than this. To get around this, I have seen other nms code create 2 nested
  loops. With the outer one striding over a small number of elements.

  See nmsLayer.cu in the TensorRT kernels.
*/
__global__ void generateBoxesKernelSimple(Prob *prob, int probWidth,
                                          int probHeight, int *outIndices,
                                          int maxOutIndices) {
  // worry about the blockIdx offset later

  // NB: Here we need the blockDim.y to be less than probWidth,
  // Otherwise we get the same threadIdx from 2 different index combinations
  int thisIdx = threadIdx.y * probWidth + threadIdx.x;
  int probSize = probWidth * probHeight;
  __shared__ int outIdx;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    outIdx = 0;
  }

  __syncthreads();

  for (int i = 0; i < probSize; i++) {
    if (thisIdx == i) {
      Prob p = prob[thisIdx];
      if (p.y > 0.5) {
        outIndices[outIdx] = thisIdx;
        printf("Gpu. thisIdx: %d. outIdx: %d\n", thisIdx, outIdx);
        outIdx++;
      }
    }
    __syncthreads();
    if (outIdx == maxOutIndices) {
      return;
    }
  }
}

// DIM is going to be the blockSize, ie blockDim.x
// I have seen it templated for loop unrolling?
template <int TSIZE, int DIM>
__global__ void generateBoxesKernel(Prob *prob, int probWidth, int probHeight,
                                    int *outIndices, int maxOutIndices) {
  // This is for a single element, ie a batch size of 1
  // I have seen nms code that uses one block per batch item
  // See nmsLayer.cu from TensorRT kernels

  Prob thisThreadProbs[TSIZE];

  __shared__ int outIdx;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    outIdx = 0;
  }

  int probSize = probWidth * probHeight;

  for (int i = 0; i < TSIZE; i++) {
    if (i * DIM + threadIdx.x < probSize) {
      thisThreadProbs[i] = prob[i * DIM + threadIdx.x];
    }
  }

  for (int i = 0; i < TSIZE; i++) {

    for (int j = 0; j < DIM; j++) {

      int offset = i * DIM;
      int index = offset + j;
      if (index >= probSize) {
        break;
      }

      __syncthreads();

      if (threadIdx.x == j) {
        Prob p = thisThreadProbs[i];
        if (p.y > 0.95) {
          outIndices[outIdx] = index;
          printf("Gpu. index: %d. outIdx: %d\n", index, outIdx);
          outIdx++;
        }
      }

      __syncthreads();

      if (outIdx == maxOutIndices) {
        return;
      }
    }
  }
}

vector<int> getIndicesAboveThreshold(const vector<Prob> &prob, int width,
                                     int height, int maxOutIndices) {
  vector<int> outIndices(maxOutIndices);
  Prob *dProb;
  int *dOutIndices;

  CUDACHECK(cudaMalloc((void **)&dProb, sizeof(Prob) * prob.size()));
  CUDACHECK(cudaMalloc((void **)&dOutIndices, sizeof(int) * outIndices.size()));

  CUDACHECK(cudaMemcpy((void *)dProb, (void *)prob.data(),
                       sizeof(Prob) * prob.size(), cudaMemcpyHostToDevice));

  int grid = 1;
  const int block = 1024;
  const int tsize = 60;

  generateBoxesKernel<tsize, block>
      <<<grid, block>>>(dProb, width, height, dOutIndices, outIndices.size());

  CUDACHECK(cudaMemcpy((void *)outIndices.data(), (void *)dOutIndices,
                       sizeof(int) * outIndices.size(),
                       cudaMemcpyDeviceToHost));

  CUDACHECK(cudaFree((void *)dProb));
  CUDACHECK(cudaFree((void *)dOutIndices));

  return outIndices;
}

int main(int argc, char **argv) {
  // int width = 2;
  // int height = 2;
  // vector<Prob> prob{{0.1, 0.9}, {0.8, 0.2}, {0.4, 0.6}, {0.3, 0.7}};
  // int maxOutIndices = 3;

  vector<Prob> prob;
  int maxOutIndices = 500;

  char arrayFilename[] =
      "./mtcnn-output-arrays/stage-one/prob-0.npy";
  cnpy::NpyArray arr = cnpy::npy_load(arrayFilename);
  vector<float> items = arr.as_vec<float>();
  int height = arr.shape.at(1);
  int width = arr.shape.at(2);

  // auto it = items.begin();
  assert(items.size() % 2 == 0);
  for (auto it = items.begin(); it != items.end();) {
    prob.emplace_back(*it, *(it + 1));
    advance(it, 2);
  }

  auto outIndices =
      getIndicesAboveThreshold(prob, width, height, maxOutIndices);

  for (auto &i : outIndices) {
    cout << i << endl;
  }
}
