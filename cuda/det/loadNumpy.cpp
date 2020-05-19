#include <iostream>
#include <vector>

#include "cnpy.h"
#include "common.h"

using namespace std;

#define TFLOAT float

int main(int argc, char **argv) {
  char arrayFilename[] =
      "/home/seb/code/ii/ml-source/mtcnn-output-arrays/stage-one/prob-0.npy";
  cnpy::NpyArray arr = cnpy::npy_load(arrayFilename);

  vector<TFLOAT> prob = arr.as_vec<TFLOAT>();
  int height = arr.shape.at(1);
  int width = arr.shape.at(2);

  cout << "Word size: " << arr.word_size << endl;

  vector<TFLOAT> sub{prob.begin(), prob.begin() + 10};
  for (auto &val : sub) {
    cout << val << ", ";
  }
  cout << endl;
}
