#include "nmsKernel.h"

// Question: Is it safe to cast float * to a BBox *?
// Answer: I don't think it is
struct BBox {
  float xMin;
  float yMin;
  float xMax;
  float yMax;

  __device__ BBox() {};
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
  int maxBoxIdx = boxesSize; // is this right? yes, yes it is
  int outBoxIdx = 0;
  int curBoxIdx = threadIdx.x;
  BBox curBox; // this threads box

  if (curBoxIdx < maxBoxIdx) {
    curBox = BBox{boxes[curBoxIdx], boxes[curBoxIdx + 1], boxes[curBoxIdx + 2],
                  boxes[curBoxIdx + 3]};
    keptBoxes[curBoxIdx] = true;
  } else {
    keptBoxes[curBoxIdx] = false;
  }

  int refBoxIdx = 0;

  while ((refBoxIdx < maxBoxIdx) && (outBoxIdx < outBoxesSize)) {
    BBox refBox{boxes[refBoxIdx], boxes[refBoxIdx + 1], boxes[refBoxIdx + 2],
                boxes[refBoxIdx + 3]};

    if (curBoxIdx > refBoxIdx) {
      if (calculateIou(refBox, curBox) > iouThreshold) {
        keptBoxes[curBoxIdx] = false;
      }
    } else if (curBoxIdx == refBoxIdx) {
      outBoxes[outBoxIdx] = refBox.xMin;
      outBoxes[outBoxIdx + 1] = refBox.yMin;
      outBoxes[outBoxIdx + 2] = refBox.xMax;
      outBoxes[outBoxIdx + 3] = refBox.yMax;
    }

    // Make sure the keptBoxes are in sync at this point
    __syncthreads();

    do {
      refBoxIdx += 4;
    } while (!keptBoxes[refBoxIdx] && refBoxIdx < maxBoxIdx);

    outBoxIdx += 4;
  }
}

void nmsSimple(float *boxes, size_t boxesSize, float *outBoxes,
               size_t outBoxesSize, float iouThreshold) {
  int grid = 1;
  int block = 256;
  nmsSimpleKernel<256>
      <<<grid, block>>>(boxes, boxesSize, outBoxes, outBoxesSize, iouThreshold);
}