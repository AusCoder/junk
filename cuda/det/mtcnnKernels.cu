#include "mtcnnKernels.h"
#include <cassert>
#include <cstdio>

__global__ void normalizePixelsKernel(float *image, size_t imageSize) {
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
__global__ void nmsSimpleKernel(float *boxes, size_t boxesSize, float *outBoxes,
                                size_t outBoxesSize, float iouThreshold) {
  // This runs off 1 block of size DIM
  // Assumes that the boxes are sorted desc by score

  // Max number of boxes we can nms with this is DIM
  __shared__ bool keptBoxes[DIM];
  int maxBoxIdx = boxesSize / 4;
  int outBoxIdx = 0;
  int curBoxIdx = threadIdx.x;
  BBox curBox;                 // this threads box

  if (curBoxIdx < maxBoxIdx) {
    curBox = BBox{boxes[4 * curBoxIdx], boxes[4 * curBoxIdx + 1],
                  boxes[4 * curBoxIdx + 2], boxes[4 * curBoxIdx + 3]};
    keptBoxes[curBoxIdx] = true;
  } else {
    keptBoxes[curBoxIdx] = false;
  }

  int refBoxIdx = 0;

  while ((refBoxIdx < maxBoxIdx) && (outBoxIdx < outBoxesSize)) {
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
cropResizeHWCKernel(const float *image, int imageWidth, int imageHeight,
                    int depth, const float *boxes, int boxesSize, int cropWidth,
                    int cropHeight, float *croppedResizedImages,
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

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                // cropHeight * depth
) {
  const int block = 1024;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeHWCKernel<<<grid, block>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t *stream) {
  const int block = 1024;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeHWCKernel<<<grid, block, 0, *stream>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

// DIM is going to be the blockSize, ie blockDim.x
// I have seen it templated for loop unrolling?
template <int TSIZE, int DIM>
__global__ void generateBoxesWithoutSoftmaxKernel(float *prob, int probSize,
                                    int *outIndices, int maxOutIndices) {
  // This is for a single element, ie a batch size of 1
  // I have seen nms code that uses one block per batch item
  // See nmsLayer.cu from TensorRT kernels

  // Prob thisThreadProbs[TSIZE];
  float thisThreadProbs[2 * TSIZE];

  __shared__ int outIdx;
  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    outIdx = 0;
  }

  // int probSize = probWidth * probHeight;

  for (int i = 0; i < TSIZE; i++) {
    // if (i * DIM + threadIdx.x < probSize) {
    if (i * DIM + threadIdx.x < probSize / 2) {
      // thisThreadProbs[i] = prob[i * DIM + threadIdx.x];
      thisThreadProbs[2 * i] = prob[2 * (i * DIM + threadIdx.x)];
      thisThreadProbs[2 * i + 1] = prob[2 * (i * DIM + threadIdx.x) + 1];
    }
  }

  for (int i = 0; i < TSIZE; i++) {

    for (int j = 0; j < DIM; j++) {

      int offset = i * DIM;
      int index = offset + j;
      if (index >= probSize / 2) {
        break;
      }

      __syncthreads();

      if (threadIdx.x == j) {
        // Prob p = thisThreadProbs[i];
        float p0 = thisThreadProbs[2 * i];
        float p1 = thisThreadProbs[2 * i + 1];

        float softmax = p1 / (p0 + p1);  // fix this!

        // Compute softmax bit here

        if (softmax > 0.95) {
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

// assumes prob is of size width * height * 2
void generateBoxesWithoutSoftmax(float *prob, int probSize,
  int *outIndices, int maxOutIndices) {
    int grid = 1;
    const int block = 1024;
    const int tsize = 60;

    generateBoxesWithoutSoftmaxKernel<tsize, block>
        <<<grid, block>>>(prob, probSize, outIndices, maxOutIndices);
  }