#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

struct BBox {
  float xMin;
  float yMin;
  float xMax;
  float yMax;

  BBox() = default;
  BBox(float xm, float ym, float xM, float yM): xMin{xm}, yMin{ym}, xMax{xM}, yMax{yM} {}

  string toString() {
    stringstream ss;
    ss << "BBox(" << xMin << ", " << yMin << ",  "
                  << xMax << ", " << yMax << ")";
    return ss.str();
  }
};

template<typename T>
vector<int> sortIndices(T *values, size_t size) {
  vector<int> indices(size);
  iota(indices.begin(), indices.end(), 0);

  sort(indices.begin(), indices.end(),
    [&values](int i1, int i2) { return values[i1] < values[i2]; });

  return indices;
}

float iou(const BBox &box1, const BBox &box2) {
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

void nmsCpu(BBox *boxes, float *scores, size_t boxesSize, BBox *outBoxes, size_t outBoxesSize, float iouThreshold) {
  vector<int> indices = sortIndices(scores, boxesSize);
  reverse(indices.begin(), indices.end());
  vector<int> keptBoxes(boxesSize);
  fill(keptBoxes.begin(), keptBoxes.end(), 1);

  size_t outBoxIdx = 0;
  size_t curBoxIdx = 0;
  while ((curBoxIdx < boxesSize) && (outBoxIdx < outBoxesSize)) {
    BBox curBox = boxes[indices[curBoxIdx]];

    outBoxes[outBoxIdx] = curBox;

    for(size_t i = outBoxIdx + 1; i < boxesSize; i++) {
      if (iou(curBox, boxes[indices[i]]) > iouThreshold) {
        keptBoxes[i] = 0;
      }
    }

    do {
      curBoxIdx++;
    } while (!keptBoxes[curBoxIdx] && (curBoxIdx < boxesSize));

    outBoxIdx++;
  }
}
