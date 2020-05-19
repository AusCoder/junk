#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

#define CUDACHECK(status)                                                      \
  do {                                                                         \
    if (status != 0) {                                                         \
      cerr << "CUDA_FAIL: " << cudaGetErrorString(status) << endl;             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

struct BBox {
  float xMin;
  float yMin;
  float xMax;
  float yMax;

  BBox() = default;
  BBox(float xm, float ym, float xM, float yM)
      : xMin{xm}, yMin{ym}, xMax{xM}, yMax{yM} {}

  string toString() {
    stringstream ss;
    ss << "BBox(" << xMin << ", " << yMin << ",  " << xMax << ", " << yMax
       << ")";
    return ss.str();
  }
};

template <typename T> vector<int> sortIndices(T *values, size_t size) {
  vector<int> indices(size);
  iota(indices.begin(), indices.end(), 0);

  sort(indices.begin(), indices.end(),
       [&values](int i1, int i2) { return values[i1] < values[i2]; });

  return indices;
}

__device__ float iou(const BBox &box1, const BBox &box2) {
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
__global__ void nmsKernelSimple(BBox *boxes, size_t boxesSize, BBox *outBoxes,
                                size_t outBoxesSize, float iouThreshold) {
  // This runs off 1 block of size DIM
  // Assumes that the boxes are sorted desc by score

  // Max number of boxes we can nms with this is DIM
  __shared__ bool keptBoxes[DIM];
  int maxBoxIdx = boxesSize;
  int outBoxIdx = 0;
  int curBoxIdx = threadIdx.x;
  BBox curBox; // this threads box

  if (curBoxIdx < maxBoxIdx) {
    curBox = boxes[curBoxIdx];
    keptBoxes[curBoxIdx] = true;
  } else {
    keptBoxes[curBoxIdx] = false;
  }

  int refBoxIdx = 0;

  while ((refBoxIdx < maxBoxIdx) && (outBoxIdx < outBoxesSize)) {
    BBox refBox = boxes[refBoxIdx];

    if (curBoxIdx > refBoxIdx) {
      if (iou(refBox, curBox) > iouThreshold) {
        keptBoxes[curBoxIdx] = false;
      }
    } else if (curBoxIdx == refBoxIdx) {
      outBoxes[outBoxIdx] = refBox;
    }

    // Make sure the keptBoxes are in sync at this point
    __syncthreads();

    do {
      refBoxIdx++;
    } while (!keptBoxes[refBoxIdx] && refBoxIdx < maxBoxIdx);

    outBoxIdx++;
  }
}

int main(int argc, char **argv) {
  float iouThreshold = 0.5;

  vector<BBox> boxes;
  boxes.emplace_back(0.1, 0.1, 0.2, 0.2);
  boxes.emplace_back(0.1, 0.1, 0.21, 0.21);
  boxes.emplace_back(0.2, 0.2, 0.3, 0.3);
  vector<float> scores{0.9, 0.8, 0.7};

  vector<BBox> outBoxes(2);

  BBox *dBoxes;
  BBox *dOutBoxes;
  CUDACHECK(cudaMalloc((void **)&dBoxes, sizeof(BBox) * boxes.size()));
  CUDACHECK(cudaMalloc((void **)&dOutBoxes, sizeof(BBox) * outBoxes.size()));

  CUDACHECK(cudaMemcpy((void *)dBoxes, (void *)boxes.data(),
                       sizeof(BBox) * boxes.size(), cudaMemcpyHostToDevice));

  int grid = 1;
  int block = 256;
  nmsKernelSimple<256><<<grid, block>>>(dBoxes, boxes.size(), dOutBoxes,
                                        outBoxes.size(), iouThreshold);
  CUDACHECK(cudaMemcpy((void *)outBoxes.data(), (void *)dOutBoxes,
                       sizeof(BBox) * outBoxes.size(), cudaMemcpyDeviceToHost));

  CUDACHECK(cudaFree((void *)dBoxes));
  CUDACHECK(cudaFree((void *)dOutBoxes));

  for (auto &box : outBoxes) {
    cout << box.toString() << endl;
  }
}
