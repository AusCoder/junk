#include <iostream>
#include <vector>
#include "nmsCpu.h"

using namespace std;

int main(int argc, char **argv) {
  vector<BBox> boxes;
  boxes.emplace_back(0.1, 0.1, 0.2, 0.2);
  boxes.emplace_back(0.1, 0.1, 0.21, 0.21);
  boxes.emplace_back(0.2, 0.2, 0.3, 0.3);
  vector<float> scores {0.9, 0.91, 0.91};

  vector<BBox> outBoxes(2);

  float iouThreshold = 0.5;
  nmsCpu(
    boxes.data(), scores.data(), boxes.size(), outBoxes.data(), outBoxes.size(), iouThreshold
  );

  for (auto &box : outBoxes) {
    cout << box.toString() << endl;
  }
}
