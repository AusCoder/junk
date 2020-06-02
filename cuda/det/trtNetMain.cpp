// TODO: Remove this program, it is just for debugging purposes
#include "cnpy.h"
#include "commonCuda.h"
#include "trtNet.h"

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  TrtNet net;

  net.start();
  int imageSize = dimsSize(net.getInputShape());
  int outputProbSize = dimsSize(net.getOutputProbShape());
  int outputRegSize = dimsSize(net.getOutputRegShape());

  // cnpy::NpyArray arr =
  // cnpy::npy_load("data/test-net-input-output/pnet_1-384-216-3_input.npy");

  // std::vector<float> image = arr.as_vec<float>();
  std::vector<float> image(imageSize);
  std::fill(image.begin(), image.end(), 0.0f);
  std::vector<float> outputProb(outputProbSize);
  std::vector<float> outputReg(outputRegSize);

  std::cout << "first values for image: ";
  for (int i = 0; i < 10; i++) {
    std::cout << image[i] << ", ";
  }
  std::cout << std::endl;

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  net.predict(image.data(), imageSize, outputProb.data(), outputProbSize,
              outputReg.data(), outputRegSize, &stream);

  std::cout << "outputProb: ";
  for (int i = 0; i < 10; i++) {
    std::cout << outputProb[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "outputReg: ";
  for (int i = 0; i < 10; i++) {
    std::cout << outputReg[i] << " ";
  }
  std::cout << std::endl;
}
