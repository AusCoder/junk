#include "mtcnnKernels.h"
#include <cassert>
#include <cstdio>
#include <cub/cub.cuh>

__global__ void normalizePixelsKernel(DevicePtr<float> image) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < image.size()) {
    image[i] -= 127.5;
    image[i] /= 128.0;
  }
}

void normalizePixels(DevicePtr<float> image, cudaStream_t &stream) {
  const int block = 1024;
  const int grid = (image.size() + block - 1) / block;
  // TODO: needs CUDACHECK?
  normalizePixelsKernel<<<grid, block, 0, stream>>>(image);
}

__global__ void denormalizePixelsKernel(float *image, size_t imageSize) {
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

__global__ void debugPrintValsKernel(float *image, size_t numVals,
                                     size_t offset) {
  int i = threadIdx.x;
  printf("%d: %f\n", i, image[offset + i]);
}

void debugPrintVals(float *image, size_t numVals, size_t offset,
                    cudaStream_t *stream) {
  assert(numVals < 1024);
  const int block = numVals;
  const int grid = 1;
  debugPrintValsKernel<<<grid, block, 0, *stream>>>(image, numVals, offset);
}

__global__ void debugPrintValsKernel(int *values, size_t numVals,
                                     size_t offset) {
  int i = threadIdx.x;
  printf("%d: %d\n", i, values[offset + i]);
}

void debugPrintVals(int *values, size_t numVals, size_t offset,
                    cudaStream_t *stream) {
  assert(numVals < 1024);
  const int block = numVals;
  const int grid = 1;
  debugPrintValsKernel<<<grid, block, 0, *stream>>>(values, numVals, offset);
}

// Question: Is it safe to cast float * to a BBox *?
// Answer: I don't think it is
struct BBox {
  float xMin;
  float yMin;
  float xMax;
  float yMax;

  __device__ BBox(){};
  __device__ BBox(float xm, float ym, float xM, float yM)
      : xMin{xm}, yMin{ym}, xMax{xM}, yMax{yM} {}
};

__device__ float calculateIou(const BBox &box1, const BBox &box2) {
  auto left = max(box1.xMin, box2.xMin);
  auto top = max(box1.yMin, box2.yMin);
  auto right = min(box1.xMax, box2.xMax);
  auto bottom = min(box1.yMax, box2.yMax);

  auto width = max(0.0f, right - left);
  auto height = max(0.0f, bottom - top);

  auto intersection = width * height;
  auto area1 = (box1.xMax - box1.xMin) * (box1.yMax - box1.yMin);
  auto area2 = (box2.xMax - box2.xMin) * (box2.yMax - box2.yMin);

  return intersection / (area1 + area2 - intersection);
}

template <int DIM>
__launch_bounds__(DIM) __global__
    void nmsSimpleKernel(float *boxes, size_t boxesSize, float *outBoxes,
                         size_t outBoxesSize, float iouThreshold) {
  // This runs off 1 block of size DIM
  // Assumes that the boxes are sorted desc by score
  // inpractice we need to sort the boxes by probability score

  // Max number of boxes we can nms with this is DIM
  __shared__ bool keptBoxes[DIM];
  int maxBoxIdx = boxesSize / 4;
  int outBoxIdx = 0;
  int curBoxIdx = threadIdx.x;
  BBox curBox; // this threads box

  if (curBoxIdx < maxBoxIdx) {
    curBox = BBox{boxes[4 * curBoxIdx], boxes[4 * curBoxIdx + 1],
                  boxes[4 * curBoxIdx + 2], boxes[4 * curBoxIdx + 3]};
    keptBoxes[curBoxIdx] = true;
  } else {
    keptBoxes[curBoxIdx] = false;
  }

  int refBoxIdx = 0;

  while ((refBoxIdx < maxBoxIdx) && (outBoxIdx < outBoxesSize / 4)) {
    BBox refBox{boxes[4 * refBoxIdx], boxes[4 * refBoxIdx + 1],
                boxes[4 * refBoxIdx + 2], boxes[4 * refBoxIdx + 3]};

    if (curBoxIdx > refBoxIdx) {
      if (calculateIou(refBox, curBox) > iouThreshold) {
        keptBoxes[curBoxIdx] = false;
      }
    } else if (curBoxIdx == refBoxIdx) {
      outBoxes[4 * outBoxIdx] = refBox.xMin;
      outBoxes[4 * outBoxIdx + 1] = refBox.yMin;
      outBoxes[4 * outBoxIdx + 2] = refBox.xMax;
      outBoxes[4 * outBoxIdx + 3] = refBox.yMax;
    }

    // Make sure the keptBoxes are in sync at this point
    __syncthreads();

    do {
      refBoxIdx += 1;
    } while (!keptBoxes[refBoxIdx] && refBoxIdx < maxBoxIdx);

    outBoxIdx += 1;
  }
}

void nmsSimple(float *boxes, size_t boxesSize, float *outBoxes,
               size_t outBoxesSize, float iouThreshold) {
  int grid = 1;
  int block = 256;
  nmsSimpleKernel<256>
      <<<grid, block>>>(boxes, boxesSize, outBoxes, outBoxesSize, iouThreshold);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void sortProb(DevicePtr<float> prob, DevicePtr<int> size, DevicePtr<int> outIndices) {
  typedef cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoad;
  typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockRadixSort;
  typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockStore;

  __shared__ union TempStorage {
    typename BlockLoad::TempStorage keys;
    typename BlockRadixSort::TempStorage sort;
    typename BlockStore::TempStorage store;
  } tempStorage;

  float threadKeys[ITEMS_PER_THREAD];
  int threadValues[ITEMS_PER_THREAD];
  BlockLoad(tempStorage.keys).Load(prob.get(), threadKeys, *(size.get()), -1);

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    threadValues[i] = threadIdx.x * ITEMS_PER_THREAD + i;
  }
  __syncthreads();
  BlockRadixSort(tempStorage.sort).SortDescending(threadKeys, threadValues);
  __syncthreads();
  BlockStore(tempStorage.store)
      .Store(outIndices.get(), threadValues, outIndices.size());
}

template <int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS) __global__
    void nmsKernel(DevicePtr<float> prob, DevicePtr<float> reg,
                   DevicePtr<float> boxes, DevicePtr<int> boxesCount,
                   DevicePtr<int> sortIndices, DevicePtr<float> outProb,
                   DevicePtr<float> outReg, DevicePtr<float> outBoxes,
                   DevicePtr<int> outBoxesPosition, float iouThreshold) {
  // rename maxBoxIdx to boxIdxsSize? It's not an index, it's a size
  // int maxBoxIdx = boxes.size() / 4;
  int boxesCount_ = *(boxesCount.get());
  size_t outProbSize = outProb.size();
  assert(boxesCount_ <= BLOCK_THREADS);
  assert(outReg.size() == 4 * outProbSize);
  assert(outBoxes.size() == 4 * outProbSize);

  __shared__ bool keptBoxes[BLOCK_THREADS];

  // int curBoxIdx = threadIdx.x;
  // int curBoxIdx = sortIndices[threadIdx.x];
  int curBoxSortIdx = threadIdx.x;
  int curBoxIdx = sortIndices[curBoxSortIdx];
  BBox curBox; // this threads box

  if (curBoxSortIdx < boxesCount_) {
    curBox = BBox{boxes[4 * curBoxIdx], boxes[4 * curBoxIdx + 1],
                  boxes[4 * curBoxIdx + 2], boxes[4 * curBoxIdx + 3]};
    keptBoxes[curBoxSortIdx] = true;
  } else {
    keptBoxes[curBoxSortIdx] = false;
  }

  // TODO: we want to offset here, so change to be *(outBoxesCount.get())
  int outIdx = *(outBoxesPosition.get());
  int refBoxSortIdx = 0;

  while ((refBoxSortIdx < boxesCount_) && (outIdx < outProbSize)) {
    int refBoxIdx = sortIndices[refBoxSortIdx];
    BBox refBox{boxes[4 * refBoxIdx + 0], boxes[4 * refBoxIdx + 1],
                boxes[4 * refBoxIdx + 2], boxes[4 * refBoxIdx + 3]};

    if (curBoxSortIdx > refBoxSortIdx) {
      if (calculateIou(refBox, curBox) > iouThreshold) {
        keptBoxes[curBoxSortIdx] = false;
      }
    } else if (curBoxSortIdx == refBoxSortIdx) {
      outProb[outIdx] = prob[refBoxIdx];
      outReg[4 * outIdx + 0] = reg[4 * refBoxIdx + 0];
      outReg[4 * outIdx + 1] = reg[4 * refBoxIdx + 1];
      outReg[4 * outIdx + 2] = reg[4 * refBoxIdx + 2];
      outReg[4 * outIdx + 3] = reg[4 * refBoxIdx + 3];
      outBoxes[4 * outIdx + 0] = refBox.xMin;
      outBoxes[4 * outIdx + 1] = refBox.yMin;
      outBoxes[4 * outIdx + 2] = refBox.xMax;
      outBoxes[4 * outIdx + 3] = refBox.yMax;

      // outIndices[outIdx] = refboxIdx;

      // outBoxes[4 * outIdx] = refBox.xMin;
      // outBoxes[4 * outIdx + 1] = refBox.yMin;
      // outBoxes[4 * outIdx + 2] = refBox.xMax;
      // outBoxes[4 * outIdx + 3] = refBox.yMax;
    }

    // Make sure the keptBoxes are in sync at this point
    __syncthreads();

    do {
      refBoxSortIdx += 1;
    } while (!keptBoxes[refBoxSortIdx] && refBoxSortIdx < boxesCount_);

    outIdx += 1;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    printf("Nms outIdx %d\n", outIdx);
    outBoxesPosition[0] = outIdx;
  }
}

void nms(DevicePtr<float> prob, DevicePtr<float> reg, DevicePtr<float> boxes,
         DevicePtr<int> boxesCount, DevicePtr<int> sortIndices,
         DevicePtr<float> outProb, DevicePtr<float> outReg,
         DevicePtr<float> outBoxes, DevicePtr<int> outBoxesPosition,
         float iouThreshold, cudaStream_t &stream) {
  sortProb<128, 8><<<1, 128, 0, stream>>>(prob, boxesCount, sortIndices);
  nmsKernel<512><<<1, 512, 0, stream>>>(prob, reg, boxes, boxesCount,
                                        sortIndices, outProb, outReg, outBoxes,
                                        outBoxesPosition, iouThreshold);
}
// blockSize will be {1024, 1, 1}
// gridSize...
__global__ void
cropResizeCHWKernel(const float *image, int imageWidth, int imageHeight,
                    int depth, const float *boxes, int boxesSize, int cropWidth,
                    int cropHeight, float *croppedResizedImages,
                    int croppedResizedImagesSize // == boxesSize * depth *
                                                 // cropWidth * cropHeight
) {
  // Assume that the depth is 3 for now

  const float extrapolationValue = 0.0f;
  const int batch = 1;

  // Each thread will loop and write to certain Idx in croppedResizedImages
  for (int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
       outIdx < croppedResizedImagesSize; outIdx += blockDim.x * gridDim.x) {
    int idx = outIdx;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int depthIdx = idx % depth;
    const int boxIdx = idx / depth;

    const float y1 = boxes[boxIdx * 4];
    const float x1 = boxes[boxIdx * 4 + 1];
    const float y2 = boxes[boxIdx * 4 + 2];
    const float x2 = boxes[boxIdx * 4 + 3];

    const int batchIdx = boxIdx / boxesSize;
    if (batchIdx < 0 || batchIdx >= batch) {
      continue;
    }

    const float heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const float widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    const float inY = (cropHeight > 1)
                          ? y1 * (imageHeight - 1) + y * heightScale
                          : 0.5 * (y1 + y2) * (imageHeight - 1);
    if (inY < 0 || inY > imageHeight - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
    if (inX < 0 || inX > imageWidth - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const int topYIndex = floorf(inY);
    const int bottomYIndex = ceilf(inY);
    const float yLerp = inY - topYIndex;
    const int leftXIndex = floorf(inX);
    const int rightXIndex = ceilf(inX);
    const float xLerp = inX - leftXIndex;

    const float topLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              leftXIndex]));
    const float topRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              rightXIndex]));
    const float bottomLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              leftXIndex]));
    const float bottomRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              rightXIndex]));

    const float top = topLeft + (topRight - topLeft) * xLerp;
    const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
    croppedResizedImages[outIdx] = top + (bottom - top) * yLerp;
  }
}

__global__ void
cropResizeHWCKernel(DevicePtr<float> image, int imageWidth, int imageHeight,
                    int depth, DevicePtr<float> boxes, int boxesSize,
                    int cropWidth, int cropHeight,
                    DevicePtr<float> croppedResizedImages,
                    int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                 // cropHeight * depth
) {
  // Assume that the depth is 3 for now

  const float extrapolationValue = 0.0f;
  const int batch = 1;

  // Each thread will loop and write to certain Idx in croppedResizedImages
  for (int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
       outIdx < croppedResizedImagesSize; outIdx += blockDim.x * gridDim.x) {
    // int idx = outIdx;
    // const int x = idx % cropWidth;
    // idx /= cropWidth;
    // const int y = idx % cropHeight;
    // idx /= cropHeight;
    // const int depthIdx = idx % depth;
    // const int boxIdx = idx / depth;
    int idx = outIdx;
    const int depthIdx = idx % depth;
    idx /= depth;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int boxIdx = idx;

    const float y1 = boxes[boxIdx * 4];
    const float x1 = boxes[boxIdx * 4 + 1];
    const float y2 = boxes[boxIdx * 4 + 2];
    const float x2 = boxes[boxIdx * 4 + 3];

    const int batchIdx = boxIdx / boxesSize;
    if (batchIdx < 0 || batchIdx >= batch) {
      continue;
    }

    const float heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const float widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    const float inY = (cropHeight > 1)
                          ? y1 * (imageHeight - 1) + y * heightScale
                          : 0.5 * (y1 + y2) * (imageHeight - 1);
    if (inY < 0 || inY > imageHeight - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
    if (inX < 0 || inX > imageWidth - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const int topYIndex = floorf(inY);
    const int bottomYIndex = ceilf(inY);
    const float yLerp = inY - topYIndex;
    const int leftXIndex = floorf(inX);
    const int rightXIndex = ceilf(inX);
    const float xLerp = inX - leftXIndex;

    const float topLeft(static_cast<float>(
        image[((batchIdx * imageHeight + topYIndex) * imageWidth + leftXIndex) *
                  depth +
              depthIdx]));
    const float topRight(static_cast<float>(
        image[((batchIdx * imageHeight + topYIndex) * imageWidth +
               rightXIndex) *
                  depth +
              depthIdx]));
    const float bottomLeft(static_cast<float>(
        image[((batchIdx * imageHeight + bottomYIndex) * imageWidth +
               leftXIndex) *
                  depth +
              depthIdx]));
    const float bottomRight(static_cast<float>(
        image[((batchIdx * imageHeight + bottomYIndex) * imageWidth +
               rightXIndex) *
                  depth +
              depthIdx]));

    const float top = topLeft + (topRight - topLeft) * xLerp;
    const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
    croppedResizedImages[outIdx] = top + (bottom - top) * yLerp;
  }
}

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                // cropHeight * depth
) {
  const int block = 1024;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeCHWKernel<<<grid, block>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t *stream) {
  const int block = 1024;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeCHWKernel<<<grid, block, 0, *stream>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeHWC(DevicePtr<float> image, int imageWidth, int imageHeight,
                   int depth, DevicePtr<float> boxes, int boxesSize,
                   int cropWidth, int cropHeight,
                   DevicePtr<float> croppedResizedImages,
                   int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                // cropHeight * depth
) {
  const int block = 512;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeHWCKernel<<<grid, block>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeHWC(DevicePtr<float> image, int imageWidth, int imageHeight,
                   int depth, DevicePtr<float> boxes, int boxesSize,
                   int cropWidth, int cropHeight,
                   DevicePtr<float> croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t &stream) {
  const int block = 512;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeHWCKernel<<<grid, block, 0, stream>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

#define BOXES_STRIDE 2
#define BOXES_CELL_SIZE 12

// DIM is going to be the blockSize, ie blockDim.x
// I have seen it templated for loop unrolling?
// TODO: Get rid of this
template <int TSIZE, int DIM>
__global__ void generateBoxesWithoutSoftmaxIndicesKernel(
    float *prob, int probWidth, int probHeight, int *outIndices,
    float *outBboxes, int maxOutIndices, float threshold, float scale) {
  // This is for a single element, ie a batch size of 1
  // I have seen nms code that uses one block per batch item
  // See nmsLayer.cu from TensorRT kernels

  // Prob thisThreadProbs[TSIZE];
  float thisThreadProbs[2 * TSIZE];

  __shared__ int outIdx;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    outIdx = 0;
  }

  int probSize = probWidth * probHeight;

  for (int i = 0; i < TSIZE; i++) {
    // if (i * DIM + threadIdx.x < probSize) {
    if (i * DIM + threadIdx.x < probSize) {
      // thisThreadProbs[i] = prob[i * DIM + threadIdx.x];
      thisThreadProbs[2 * i] = prob[2 * (i * DIM + threadIdx.x)];
      thisThreadProbs[2 * i + 1] = prob[2 * (i * DIM + threadIdx.x) + 1];
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
        // Prob p = thisThreadProbs[i];
        float p0 = thisThreadProbs[2 * i];
        float p1 = thisThreadProbs[2 * i + 1];

        p0 = expf(p0);
        p1 = expf(p1);
        float softmax = p1 / (p0 + p1);

        if (softmax > threshold) {
          int x = index % probWidth;
          int y = index / probWidth;
          outIndices[outIdx] = index;
          outBboxes[4 * outIdx + 0] = (x * BOXES_STRIDE + 1) / scale;
          outBboxes[4 * outIdx + 1] = (y * BOXES_STRIDE + 1) / scale;
          outBboxes[4 * outIdx + 2] =
              (x * BOXES_STRIDE + BOXES_CELL_SIZE) / scale;
          outBboxes[4 * outIdx + 3] =
              (y * BOXES_STRIDE + BOXES_CELL_SIZE) / scale;
          // printf("Gpu. index: %d. outIdx: %d\n", index, outIdx);
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

void generateBoxesWithoutSoftmaxIndices(float *prob, int probWidth,
                                        int probHeight, int *outIndices,
                                        float *outBboxes, int maxOutIndices,
                                        float threshold, float scale,
                                        cudaStream_t *stream) {
  int grid = 1;
  const int block = 1024;
  const int tsize = 60;

  generateBoxesWithoutSoftmaxIndicesKernel<tsize, block>
      <<<grid, block, 0, *stream>>>(prob, probWidth, probHeight, outIndices,
                                    outBboxes, maxOutIndices, threshold, scale);
}

// DIM is going to be the blockSize, ie blockDim.x
// I have seen it templated for loop unrolling?
template <int TSIZE, int DIM>
__global__ void generateBoxesWithoutSoftmaxKernel(
    float *prob, int probWidth, int probHeight, float *reg, int regWidth,
    int regHeight, float *outProb, float *outReg, float *outBboxes,
    int *outBoxesCount, int maxOutputBoxes, float threshold, float scale) {
  // This is for a single element, ie a batch size of 1
  // I have seen nms code that uses one block per batch item
  // See nmsLayer.cu from TensorRT kernels

  float thisThreadProbs[2 * TSIZE];

  __shared__ int outIdx;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    outIdx = 0;
  }

  assert(probWidth == regWidth);
  assert(probHeight == regHeight);
  int probSize = probWidth * probHeight;

  for (int i = 0; i < TSIZE; i++) {
    if (i * DIM + threadIdx.x < probSize) {
      thisThreadProbs[2 * i] = prob[2 * (i * DIM + threadIdx.x)];
      thisThreadProbs[2 * i + 1] = prob[2 * (i * DIM + threadIdx.x) + 1];
    }
  }

  bool shouldBreak = false;

  for (int i = 0; i < TSIZE; i++) {
    for (int j = 0; j < DIM; j++) {
      int offset = i * DIM;
      int index = offset + j;
      if (index >= probSize) {
        break;
      }

      __syncthreads();

      if (threadIdx.x == j) {
        // Prob p = thisThreadProbs[i];
        float p0 = thisThreadProbs[2 * i];
        float p1 = thisThreadProbs[2 * i + 1];

        p0 = expf(p0);
        p1 = expf(p1);
        float softmax = p1 / (p0 + p1);

        if (softmax > threshold) {
          int x = index % probWidth;
          int y = index / probWidth;
          // filter prob and reg
          outProb[outIdx] = softmax;
          outReg[4 * outIdx + 0] = reg[4 * outIdx + 0];
          outReg[4 * outIdx + 1] = reg[4 * outIdx + 1];
          outReg[4 * outIdx + 2] = reg[4 * outIdx + 2];
          outReg[4 * outIdx + 3] = reg[4 * outIdx + 3];
          // create output bounding box
          outBboxes[4 * outIdx + 0] = (x * BOXES_STRIDE + 1) / scale;
          outBboxes[4 * outIdx + 1] = (y * BOXES_STRIDE + 1) / scale;
          outBboxes[4 * outIdx + 2] =
              (x * BOXES_STRIDE + BOXES_CELL_SIZE) / scale;
          outBboxes[4 * outIdx + 3] =
              (y * BOXES_STRIDE + BOXES_CELL_SIZE) / scale;
          // printf("Gpu. index: %d. outIdx: %d\n", index, outIdx);
          outIdx++;
        }
      }

      __syncthreads();

      if (outIdx == maxOutputBoxes) {
        shouldBreak = true;
        break;
        // return;
      }
    }
    if (shouldBreak) {
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("Generate boxes outIdx %d\n", outIdx);
    outBoxesCount[0] = outIdx;
  }
}

void generateBoxesWithoutSoftmax(float *prob, int probWidth, int probHeight,
                                 float *reg, int regWidth, int regHeight,
                                 float *outProb, float *outReg,
                                 float *outBboxes, int *outBoxesCount,
                                 int maxOutputBoxes, float threshold,
                                 float scale, cudaStream_t *stream) {
  const int grid = 1;
  const int block = 1024;
  const int tsize = 60;
  generateBoxesWithoutSoftmaxKernel<tsize, block><<<grid, block, 0, *stream>>>(
      prob, probWidth, probHeight, reg, regWidth, regHeight, outProb, outReg,
      outBboxes, outBoxesCount, maxOutputBoxes, threshold, scale);
}

template <int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS) __global__
    void regressAndSquareBoxesKernel(DevicePtr<float> boxes,
                                     DevicePtr<float> reg,
                                     DevicePtr<int> boxesCount,
                                     bool shouldSquare) {
  int boxesCount_ = *(boxesCount.get());
  assert(boxesCount_ <= BLOCK_THREADS);

  int boxIdx = threadIdx.x;
  if (boxIdx < boxesCount_) {
    float width = boxes[4 * boxIdx + 2] - boxes[4 * boxIdx + 0];
    float height = boxes[4 * boxIdx + 3] - boxes[4 * boxIdx + 1];
    boxes[4 * boxIdx + 0] += width * reg[4 * boxIdx + 0];
    boxes[4 * boxIdx + 1] += height * reg[4 * boxIdx + 1];
    boxes[4 * boxIdx + 2] += width * reg[4 * boxIdx + 2];
    boxes[4 * boxIdx + 3] += height * reg[4 * boxIdx + 3];

    if (shouldSquare) {
      width = boxes[4 * boxIdx + 2] - boxes[4 * boxIdx + 0];
      height = boxes[4 * boxIdx + 3] - boxes[4 * boxIdx + 1];
      float maxSide = max(width, height);
      float deltaWidth = (width - maxSide) / 2;
      float deltaHeight = (height - maxSide) / 2;
      boxes[4 * boxIdx + 0] += deltaWidth;
      boxes[4 * boxIdx + 1] += deltaHeight;
      boxes[4 * boxIdx + 2] -= deltaWidth;
      boxes[4 * boxIdx + 3] -= deltaHeight;
    }
  }
}

void regressAndSquareBoxes(DevicePtr<float> boxes, DevicePtr<float> reg,
                           DevicePtr<int> boxesCount, bool shouldSquare,
                           cudaStream_t &stream) {
  const int grid = 1;
  const int block = 512;
  regressAndSquareBoxesKernel<block>
      <<<grid, block, 0, stream>>>(boxes, reg, boxesCount, shouldSquare);
}

/**
 * NB: We cannot pass references to a kernel, they need to be passed by value
 * It makes sense because a reference would exist on Host memory and cannot be
 * accessed from the device.
 */
__global__ void scalarMultKernel(DevicePtr<float> p, size_t pSize,
                                 float value) {
  if (threadIdx.x < pSize) {
    p[threadIdx.x] *= value;
  }
}

void scalarMult(DevicePtr<float> p, size_t pSize, float value,
                cudaStream_t &stream) {
  scalarMultKernel<<<1, 128, 0, stream>>>(p, pSize, value);
}