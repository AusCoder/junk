// TODO: Remove this program, it is just for debugging purposes
#include "cnpy.h"
#include "commonCuda.h"
#include "trtNet.h"

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  TrtNetInfo pnetInfo{TrtNet::createPnetInfo()};
  TrtNet net{"data/debug_uff/debug_net.uff", pnetInfo};
  net.start();

  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> outputs;

  inputs.emplace_back(dimsSize(pnetInfo.inputTensorInfos[0].shape));
  assert(inputs[0].capacity() == dimsSize(pnetInfo.inputTensorInfos[0].shape));
  outputs.emplace_back(dimsSize(pnetInfo.outputTensorInfos[0].shape));
  outputs.emplace_back(dimsSize(pnetInfo.outputTensorInfos[1].shape));

  std::fill(inputs[0].begin(), inputs[0].end(), 0.0f);

  // int imageSize = dimsSize(net.getInputShape());
  // int outputProbSize = dimsSize(net.getOutputProbShape());
  // int outputRegSize = dimsSize(net.getOutputRegShape());

  // cnpy::NpyArray arr =
  // cnpy::npy_load("data/test-net-input-output/pnet_1-384-216-3_input.npy");

  // std::vector<float> image = arr.as_vec<float>();
  // std::vector<float> image(imageSize);
  // std::fill(image.begin(), image.end(), 0.0f);
  // std::vector<float> outputProb(outputProbSize);
  // std::vector<float> outputReg(outputRegSize);

  std::cout << "first values for image: ";
  for (int i = 0; i < 10; i++) {
    std::cout << inputs[0][i] << ", ";
  }
  std::cout << std::endl;

  std::vector<float *> netInputs;
  for (auto &input : inputs) {
    netInputs.push_back(input.data());
  }
  std::vector<float *> netOutputs;
  for (auto &output : outputs) {
    netOutputs.push_back(output.data());
  }

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // net.predict(image.data(), imageSize, outputProb.data(), outputProbSize,
  //             outputReg.data(), outputRegSize, &stream);
  net.predictFromHost(netInputs, netOutputs, &stream);

  std::cout << "outputProb: ";
  for (int i = 0; i < 10; i++) {
    std::cout << outputs[0][i] << " ";
  }
  std::cout << std::endl;

  std::cout << "outputReg: ";
  for (int i = 0; i < 10; i++) {
    std::cout << outputs[1][i] << " ";
  }
  std::cout << std::endl;

  CUDACHECK(cudaStreamDestroy(stream));
}
