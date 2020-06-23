#include "NvUffParser.h"
#include "commonCuda.h"
#include "logger.h"
#include "trtNet.h"

#include <cassert>
#include <iostream>
#include <string>

TensorInfo::TensorInfo(std::string n, nvinfer1::Dims3 s, InputOrder i):
  name{n}, shape{s}, inputOrder{i} {}

TensorInfo::TensorInfo(std::string n, nvinfer1::Dims3 s):
  name{n}, shape{s}, inputOrder{InputOrder::None} {}

TrtNet::TrtNet(const std::string &p, const TrtNetInfo &i): modelPath{p}, netInfo{i} {}

TrtNet::TrtNet(TrtNet &&net): builder{net.builder}, engine{net.engine}, context{net.context}, modelPath{net.modelPath}, netInfo{net.netInfo} {
  net.builder = nullptr;
  net.engine = nullptr;
  net.context = nullptr;
  net.modelPath = "";
  net.netInfo = {};
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

// nvinfer1::Dims3 TrtNet::getInputShape() { return inputShape; };
// nvinfer1::Dims3 TrtNet::getOutputProbShape() { return outputProbShape; };
// nvinfer1::Dims3 TrtNet::getOutputRegShape() { return outputRegShape; };
const TensorInfo &TrtNet::getInputTensorInfo(int i) {
  return netInfo.inputTensorInfos.at(i);
}

const TensorInfo &TrtNet::getOutputTensorInfo(int i) {
  return netInfo.outputTensorInfos.at(i);
}

/**
 * Can we put more than one graph in a ICudaEngine?
 */
void TrtNet::start() {
  builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  nvuffparser::IUffParser *parser = nvuffparser::createUffParser();

  for (auto &tensorInfo : netInfo.inputTensorInfos) {
    nvuffparser::UffInputOrder order;
    if (tensorInfo.inputOrder == InputOrder::NHWC) {
      order = nvuffparser::UffInputOrder::kNHWC;
    }
    // TODO: throw here
    // else {
      // throw std::
    // }
    parser->registerInput(tensorInfo.name.c_str(), tensorInfo.shape,
                        order);
  }
  for (auto &tensorInfo : netInfo.outputTensorInfos) {
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
  int numBuffers = netInfo.inputTensorInfos.size() + netInfo.outputTensorInfos.size();
  void *buffers[numBuffers];

  for (int i = 0; i < inputs.size(); i++) {
    const TensorInfo &tensorInfo = netInfo.inputTensorInfos.at(i);
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
    const TensorInfo &tensorInfo = netInfo.outputTensorInfos.at(i);
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
    const TensorInfo &tensorInfo = netInfo.outputTensorInfos.at(i);
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

TrtNetInfo TrtNet::createPnetInfo() {
  TrtNetInfo pnetInfo{};
  pnetInfo.inputTensorInfos.push_back(
      {"input_1", {384, 216, 3}, InputOrder::NHWC});
  pnetInfo.outputTensorInfos.push_back({"conv2d_3/BiasAdd", {187, 103, 2}});
  pnetInfo.outputTensorInfos.push_back({"conv2d_4/BiasAdd", {187, 103, 4}});
  return pnetInfo;
}
