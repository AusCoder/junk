#include "trtNet.h"

#include <iostream>
#include <string>

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
  parser->registerInput("input_1", nvinfer1::Dims3(384, 216, 3),
                        nvuffparser::UffInputOrder::kNHWC);
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

// void TrtNet::predict() {
//   // TODO: make an array of ones and test the network on that
// }

void TrtNet::predict(float *image, int height, int width, float *outArr,
                     int outArrSize) {
  int inputIndex = engine->getBindingIndex("input_1");
  int outputIndex1 = engine->getBindingIndex("softmax/Softmax");
  int outputIndex2 = engine->getBindingIndex("conv2d_4/BiasAdd");

  // TODO: copy image to gpu
  void *buffers[3];
  buffers[inputIndex] = inputbuffer;
  buffers[outputIndex1] = outputBuffer;
}
