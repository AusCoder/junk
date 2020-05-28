#include "trtNet.h"

#include <iostream>
#include <string>

// TODO: fix the shapes here
TrtNet::TrtNet()
    : inputName{"input_1"}, outputProbName{"softmax/Softmax"},
      outputRegName{"conv2d_4/BiasAdd"}, inputShape{384, 216, 3},
      outputProbShape{1, 1, 1}, outputRegShape{1, 1, 1} {}

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

/**
 * Can we put more than one graph in a ICudaEngine?
 */
void TrtNet::start() {
  std::string uffFile = "data/uff/pnet_216x384.uff";

  builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  nvuffparser::IUffParser *parser = nvuffparser::createUffParser();
  parser->registerInput("input_1", , nvuffparser::UffInputOrder::kNHWC);
  parser->registerOutput("softmax/Softmax");
  parser->registerOutput("conv2d_4/BiasAdd");
  parser->parse(uffFile.c_str(), *network, nvinfer1::DataType::kFLOAT);
  parser->destroy();

  builder->setMaxBatchSize(16);
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
void TrtNet::predict(float *image, int height, int width, int channels,
                     float *outArr, int outArrSize) {
  // TODO: add check for height, width, channels to match the inputShape
  int inputIndex = engine->getBindingIndex(inputName.c_str());
  int outputIndex1 = engine->getBindingIndex(outputProbName.c_str());
  int outputIndex2 = engine->getBindingIndex(outputRegName.c_str());

  float *dImage;
  float *dOutputProb;
  float *dOutputReg;

  int imageSize = inputShape.d[0] * inputShape.d[1] * inputShape.d[2];
  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * imageSize));

  CUDACHECK(cudaMemcpy((void *)dImage, (void *)image, sizeof(float) * imageSize,
                       cudaMemcpyHostToDevice));

  void *buffers[3];
  buffers[inputIndex] = dImage;
  buffers[outputIndex1] = outputBuffer;
}
