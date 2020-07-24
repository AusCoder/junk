#include "trtNet.h"
#include "NvUffParser.h"
#include "commonCuda.h"
#include "logger.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

TrtNet::TrtNet(const std::string &p, const TrtNetInfo &i)
    : modelPath{p}, trtNetInfo{i} {}

TrtNet::TrtNet(TrtNet &&net)
    : builder{net.builder}, engine{net.engine}, context{net.context},
      modelPath{net.modelPath}, trtNetInfo{net.trtNetInfo} {
  net.builder = nullptr;
  net.engine = nullptr;
  net.context = nullptr;
  net.modelPath = "";
  net.trtNetInfo = {};
}

TrtNet::~TrtNet() {
  if (context != nullptr) {
    context->destroy();
  }
  if (engine != nullptr) {
    engine->destroy();
  }
  if (builder != nullptr) {
    builder->destroy();
  }
}

const TrtNetInfo &TrtNet::getTrtNetInfo() { return trtNetInfo; }
const TensorInfo &TrtNet::getInputTensorInfo(int i) {
  return trtNetInfo.inputTensorInfos.at(i);
}

const TensorInfo &TrtNet::getOutputTensorInfo(int i) {
  return trtNetInfo.outputTensorInfos.at(i);
}

/**
 * Can we put more than one graph in a ICudaEngine?
 */
void TrtNet::start() {
  builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  nvuffparser::IUffParser *parser = nvuffparser::createUffParser();

  for (auto &tensorInfo : trtNetInfo.inputTensorInfos) {
    nvuffparser::UffInputOrder order;
    if (tensorInfo.inputOrder == TensorInputOrder::NHWC) {
      order = nvuffparser::UffInputOrder::kNHWC;
    }
    // TODO: throw here
    // else {
    // throw std::
    // }
    parser->registerInput(tensorInfo.name.c_str(), tensorInfo.shape, order);
  }
  for (auto &tensorInfo : trtNetInfo.outputTensorInfos) {
    parser->registerOutput(tensorInfo.name.c_str());
  }

  parser->parse(modelPath.c_str(), *network, nvinfer1::DataType::kFLOAT);

  builder->setMaxBatchSize(1);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 30);
  engine = builder->buildEngineWithConfig(*network, *config);

  context = engine->createExecutionContext();

  std::cout << "Created ICudaEngine" << std::endl;
  network->destroy();
  parser->destroy();
}

// Predict using device pointers
// Assumes that inputs already contains the data for prediction
void TrtNet::predict(const std::vector<float *> &inputs,
                     const std::vector<float *> &outputs, int batchSize,
                     cudaStream_t *stream) {
  int numBuffers =
      trtNetInfo.inputTensorInfos.size() + trtNetInfo.outputTensorInfos.size();
  void *buffers[numBuffers];

  for (int i = 0; i < inputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.inputTensorInfos.at(i);
    int bindingIndex = engine->getBindingIndex(tensorInfo.name.c_str());
    buffers[bindingIndex] = inputs.at(i);
  }
  for (int i = 0; i < outputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.outputTensorInfos.at(i);
    int bindingIndex = engine->getBindingIndex(tensorInfo.name.c_str());
    buffers[bindingIndex] = outputs.at(i);
  }

  if (!context->enqueue(batchSize, buffers, *stream, nullptr)) {
    std::cout << "Execute failed" << std::endl;
  } else {
    std::cout << "Execute success" << std::endl;
  }
}

// Input the sizes here as a check?
void TrtNet::predictFromHost(const std::vector<float *> &inputs,
                             const std::vector<float *> &outputs, int batchSize,
                             cudaStream_t *stream) {
  std::vector<float *> deviceInputs(inputs.size());
  std::vector<float *> deviceOutputs(outputs.size());

  for (int i = 0; i < inputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.inputTensorInfos.at(i);
    int inputSize = tensorInfo.volume();
    float *input = inputs.at(i);
    CUDACHECK(
        cudaMalloc((void **)&deviceInputs.at(i), sizeof(float) * inputSize));
    CUDACHECK(cudaMemcpyAsync((void *)deviceInputs.at(i), (void *)input,
                              sizeof(float) * inputSize, cudaMemcpyHostToDevice,
                              *stream));
  }

  for (int i = 0; i < outputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.outputTensorInfos.at(i);
    int outputSize = tensorInfo.volume();
    CUDACHECK(
        cudaMalloc((void **)&deviceOutputs.at(i), sizeof(float) * outputSize));
  }

  predict(deviceInputs, deviceOutputs, batchSize, stream);

  for (int i = 0; i < outputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.outputTensorInfos.at(i);
    int outputSize = tensorInfo.volume();
    CUDACHECK(cudaMemcpyAsync(
        (void *)outputs.at(i), (void *)deviceOutputs.at(i),
        sizeof(float) * outputSize, cudaMemcpyDeviceToHost, *stream));
  }

  CUDACHECK(cudaStreamSynchronize(*stream));

  for (int i = 0; i < deviceInputs.size(); i++) {
    CUDACHECK(cudaFree(deviceInputs[i]));
  }
  for (int i = 0; i < deviceOutputs.size(); i++) {
    CUDACHECK(cudaFree(deviceOutputs[i]));
  }
}
