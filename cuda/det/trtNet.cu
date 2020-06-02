#include "commonCuda.h"
#include "logger.h"
#include "trtNet.h"

#include <iostream>
#include <string>

TrtNetInfo::TrtNetInfo(nvinfer1::Dims3 in, nvinfer1::Dims3 outProb,
                       nvinfer1::Dims3 outReg)
    : inputShape{in}, outputProbShape{outProb}, outputRegShape{outReg} {}

TrtNetInfo::TrtNetInfo(nvinfer1::Dims3 in, nvinfer1::Dims3 outProb,
                       nvinfer1::Dims3 outReg, nvinfer1::Dims3 outLand)
    : inputShape{in}, outputProbShape{outProb}, outputRegShape{outReg},
      outputLandmarksShape{outLand} {}

std::map<std::pair<int, int>, TrtNetInfo> TrtNet::TRT_NET_INFO = {
    {{384, 216}, {{384, 216, 3}, {187, 103, 2}, {187, 103, 4}}},
};

TrtNet::TrtNet()
    : inputName{"input_1"}, outputProbName{"softmax/Softmax"},
      outputRegName{"conv2d_4/BiasAdd"},
      inputShape{TRT_NET_INFO.at({384, 216}).inputShape},
      outputProbShape{TRT_NET_INFO.at({384, 216}).outputProbShape},
      outputRegShape{TRT_NET_INFO.at({384, 216}).outputRegShape} {}

TrtNet::~TrtNet() {
  if (builder != nullptr) {
    builder->destroy();
  }
  if (engine != nullptr) {
    engine->destroy();
  }
  if (context != nullptr) {
    context->destroy();
  }
}

nvinfer1::Dims3 TrtNet::getInputShape() {
  return inputShape;
};
nvinfer1::Dims3 TrtNet::getOutputProbShape() {
  return outputProbShape;
};
nvinfer1::Dims3 TrtNet::getOutputRegShape() {
  return outputRegShape;
};

/**
 * Can we put more than one graph in a ICudaEngine?
 */
void TrtNet::start() {
  std::string uffFile = "data/uff/pnet_216x384.uff";

  builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  nvuffparser::IUffParser *parser = nvuffparser::createUffParser();
  parser->registerInput(inputName.c_str(), inputShape,
                        nvuffparser::UffInputOrder::kNHWC);
  parser->registerOutput(outputProbName.c_str());
  parser->registerOutput(outputRegName.c_str());
  parser->parse(uffFile.c_str(), *network, nvinfer1::DataType::kFLOAT);
  parser->destroy();

  builder->setMaxBatchSize(4);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 20);
  engine = builder->buildEngineWithConfig(*network, *config);
  network->destroy();

  context = engine->createExecutionContext();

  std::cout << "Created ICudaEngine" << std::endl;
}

/**
 * For now, we assume image is a cpu array
 */
void TrtNet::predict(float *image, int imageSize,
                     float *outputProb, int outputProbSize, float *outputReg,
                     int outputRegSize) {
  // TODO: add check for height, width, channels to match the inputShape
  int inputIndex = engine->getBindingIndex(inputName.c_str());
  int outputIndexProb = engine->getBindingIndex(outputProbName.c_str());
  int outputIndexReg = engine->getBindingIndex(outputRegName.c_str());

  float *dImage;
  float *dOutputProb;
  float *dOutputReg;

  assert(imageSize == dimsSize(inputShape));
  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * imageSize));
  CUDACHECK(cudaMalloc((void **)&dOutputProb,
                       sizeof(float) * dimsSize(outputProbShape)));
  CUDACHECK(cudaMalloc((void **)&dOutputReg,
                       sizeof(float) * dimsSize(outputRegShape)));

  CUDACHECK(cudaMemcpy((void *)dImage, (void *)image, sizeof(float) * imageSize,
                       cudaMemcpyHostToDevice));

  void *buffers[3];
  buffers[inputIndex] = dImage;
  buffers[outputIndexProb] = dOutputProb;
  buffers[outputIndexReg] = dOutputReg;

  context->execute(1, buffers);

  CUDACHECK(cudaMemcpy((void *)outputProb, (void *)dOutputProb,
                       sizeof(float) * outputProbSize, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy((void *)outputReg, (void *)dOutputReg,
                       sizeof(float) * outputRegSize, cudaMemcpyDeviceToHost));
}