#include "NvUffParser.h"
#include "commonCuda.h"
#include "logger.h"
#include "trtNet.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

TrtNet::TrtNet(const std::string &p, const TrtNetInfo &i): modelPath{p}, trtNetInfo{i} {}

TrtNet::TrtNet(TrtNet &&net): builder{net.builder}, engine{net.engine}, context{net.context}, modelPath{net.modelPath}, trtNetInfo{net.trtNetInfo} {
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

const TrtNetInfo &TrtNet::getTrtNetInfo() {
  return trtNetInfo;
}
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
    parser->registerInput(tensorInfo.name.c_str(), tensorInfo.shape,
                        order);
  }
  for (auto &tensorInfo : trtNetInfo.outputTensorInfos) {
    parser->registerOutput(tensorInfo.name.c_str());
  }
  // parser->registerInput(inputName.c_str(), inputShape,
  //                       nvuffparser::UffInputOrder::kNHWC);
  // parser->registerOutput(outputProbName.c_str());
  // parser->registerOutput(outputRegName.c_str());

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

// void TrtNet::predict(const std::vector<float *> inputs)

void TrtNet::predictFromHost(const std::vector<float *> &inputs, const std::vector<float *> &outputs,
  cudaStream_t *stream) {
  int numBuffers = trtNetInfo.inputTensorInfos.size() + trtNetInfo.outputTensorInfos.size();
  void *buffers[numBuffers];

  for (int i = 0; i < inputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.inputTensorInfos.at(i);
    int bindingIndex = engine->getBindingIndex(
      tensorInfo.name.c_str()
    );
    int inputSize = dimsSize(tensorInfo.shape);
    float *input = inputs.at(i);
    float *dInput;
    CUDACHECK(cudaMalloc((void **)&dInput, sizeof(float) * inputSize));

    CUDACHECK(cudaMemcpyAsync((void *)dInput, (void *)input,
                            sizeof(float) * inputSize, cudaMemcpyHostToDevice,
                            *stream));
    buffers[bindingIndex] = dInput;
  }

  for (int i = 0; i < outputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.outputTensorInfos.at(i);
    int bindingIndex = engine->getBindingIndex(
      tensorInfo.name.c_str()
    );
    int outputSize = dimsSize(tensorInfo.shape);
    float *output = outputs.at(i);
    float *dOutput;
    CUDACHECK(cudaMalloc((void **)&dOutput, sizeof(float) * outputSize));
    buffers[bindingIndex] = dOutput;
  }

  if (!context->enqueue(1, buffers, *stream, nullptr)) {
    std::cout << "Execute failed" << std::endl;
  } else {
    std::cout << "Execute success" << std::endl;
  }

  for (int i = 0; i < outputs.size(); i++) {
    const TensorInfo &tensorInfo = trtNetInfo.outputTensorInfos.at(i);
    int bindingIndex = engine->getBindingIndex(
      tensorInfo.name.c_str()
    );
    int outputSize = dimsSize(tensorInfo.shape);
    float *output = outputs.at(i);
    CUDACHECK(cudaMemcpyAsync((void *)output, buffers[bindingIndex],
                            sizeof(float) * outputSize,
                            cudaMemcpyDeviceToHost, *stream));
  }

  CUDACHECK(cudaStreamSynchronize(*stream));

  for (int i = 0; i < numBuffers; i++) {
    CUDACHECK(cudaFree(buffers[i]));
  }
}
