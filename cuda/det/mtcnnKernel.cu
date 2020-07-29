#include "mtcnnKernel.h"
#include <cassert>
#include <cstdio>

__global__ void normalizePixelsKernel(
  float *image, size_t imageSize
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < imageSize) {
    image[i] -= 127.5;
    image[i] /= 128.0;
  }
}

void normalizePixels(float *image, size_t imageSize, cudaStream_t *stream) {
  const int block = 1024;
  const int grid = (imageSize + block - 1) / block;
  // TODO: needs CUDACHECK?
  normalizePixelsKernel<<<grid, block, 0, *stream>>>(image, imageSize);
}

__global__ void denormalizePixelsKernel(
  float *image, size_t imageSize
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < imageSize) {
    image[i] *= 128.0;
    image[i] += 127.5;
  }
}

void denormalizePixels(float *image, size_t imageSize, cudaStream_t *stream) {
  const int block = 1024;
  const int grid = (imageSize + block - 1) / block;
  denormalizePixelsKernel<<<grid, block, 0, *stream>>>(image, imageSize);
}

__global__ void debugPrintValsKernel(float *image, size_t numVals, size_t offset) {
  int i = threadIdx.x;
  printf("%d: %f\n", i, image[offset + i]);
}

void debugPrintVals(float *image, size_t numVals, size_t offset, cudaStream_t *stream) {
  assert(numVals < 1024);
  const int block = numVals;
  const int grid = 1;
  debugPrintValsKernel<<<grid, block, 0, *stream>>>(image, numVals, offset);
}
