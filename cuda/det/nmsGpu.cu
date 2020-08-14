#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "commonCuda.hpp"
#include "mtcnnKernels.h"

using namespace std;

// Question: Is it safe to cast float * to a BBox *?
// Answer: I don't think it is
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

int main(int argc, char **argv) {
  float iouThreshold = 0.5;

  vector<BBox> bboxes;
  bboxes.emplace_back(0.1, 0.1, 0.2, 0.2);
  bboxes.emplace_back(0.1, 0.1, 0.21, 0.21);
  bboxes.emplace_back(0.2, 0.2, 0.3, 0.3);
  vector<float> scores{0.9, 0.8, 0.7};

  vector<float> boxes;
  std::for_each(
    bboxes.begin(), bboxes.end(),
    [&boxes](auto &box) -> void {
      boxes.push_back(box.xMin);
      boxes.push_back(box.yMin);
      boxes.push_back(box.xMax);
      boxes.push_back(box.yMax);
    }
  );

  vector<float> outBoxes(20);

  float *dBoxes;
  float *dOutBoxes;
  CUDACHECK(cudaMalloc((void **)&dBoxes, sizeof(float) * boxes.size()));
  CUDACHECK(cudaMalloc((void **)&dOutBoxes, sizeof(float) * outBoxes.size()));

  CUDACHECK(cudaMemcpy((void *)dBoxes, (void *)boxes.data(),
                       sizeof(float) * boxes.size(), cudaMemcpyHostToDevice));

  nmsSimple(dBoxes, boxes.size(), dOutBoxes, outBoxes.size(), iouThreshold);

  CUDACHECK(cudaMemcpy((void *)outBoxes.data(), (void *)dOutBoxes,
                       sizeof(float) * outBoxes.size(), cudaMemcpyDeviceToHost));

  CUDACHECK(cudaFree((void *)dBoxes));
  CUDACHECK(cudaFree((void *)dOutBoxes));

  vector<BBox> outBboxes;

  for (auto it = outBoxes.begin(); it != outBoxes.end();) {
    outBboxes.push_back({
      *(it++), *(it++), *(it++), *(it++)
    });
  }

  for (auto &box : outBboxes) {
    cout << box.toString() << endl;
  }
}
