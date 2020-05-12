/*
Kernel to crop and resize boxes from an image
*/

// blockSize will be {1024, 1, 1}
// gridSize...
__global__ void cropResizeKernel(
  const float *image, int imageWidth, int imageHeight, int depth
  const float *boxes, int boxesSize,
  int cropWidth, int cropHeight,
  float *croppedBoxes,
  int croppedBoxesSize // == boxesSize * cropWidth * cropHeight * depth
) {
  // Assume that the depth is 3 for now

  // Each thread will loop and write to certain Idx in croppedBoxes
  for (
    int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
    outIdx < croppedBoxesSize;
    outIdx += blockDim.x * gridDim.x
  ) {
    int idx = outIdx;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int depthIdx
  }
}
