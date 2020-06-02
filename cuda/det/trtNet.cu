#include "commonCuda.h"
#include "logger.h"
#include "trtNet.h"

#include <cassert>
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
    : inputName{"input_1"}, outputProbName{"conv2d_3/BiasAdd"},
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

nvinfer1::Dims3 TrtNet::getInputShape() { return inputShape; };
nvinfer1::Dims3 TrtNet::getOutputProbShape() { return outputProbShape; };
nvinfer1::Dims3 TrtNet::getOutputRegShape() { return outputRegShape; };

/**
 * Can we put more than one graph in a ICudaEngine?
 */
void TrtNet::start() {
  std::string uffFile = "data/debug_uff/debug_net.uff";

  builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  nvuffparser::IUffParser *parser = nvuffparser::createUffParser();
  parser->registerInput(inputName.c_str(), inputShape,
                        nvuffparser::UffInputOrder::kNHWC);
  parser->registerOutput(outputProbName.c_str());
  parser->registerOutput(outputRegName.c_str());
  parser->parse(uffFile.c_str(), *network, nvinfer1::DataType::kFLOAT);

  builder->setMaxBatchSize(1);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 30);
  engine = builder->buildEngineWithConfig(*network, *config);

  context = engine->createExecutionContext();

  std::cout << "Created ICudaEngine" << std::endl;
  network->destroy();
  parser->destroy();
}

/**
 * For now, we assume image is a cpu array
 */
void TrtNet::predict(float *image, int imageSize, float *outputProb,
                     int outputProbSize, float *outputReg, int outputRegSize,
                     cudaStream_t *stream) {

  // TODO: add check for height, width, channels to match the inputShape
  int inputIndex = engine->getBindingIndex(inputName.c_str());
  int outputIndexProb = engine->getBindingIndex(outputProbName.c_str());
  int outputIndexReg = engine->getBindingIndex(outputRegName.c_str());

  std::cout << "Binding indices: " << inputIndex << " " << outputIndexProb
            << std::endl;

  float *dImage;
  float *dOutputProb;
  float *dOutputReg;

  assert(imageSize == dimsSize(inputShape));
  assert(outputProbSize == dimsSize(outputProbShape));
  assert(outputRegSize == dimsSize(outputRegShape));

  CUDACHECK(cudaMalloc((void **)&dImage, sizeof(float) * imageSize));
  CUDACHECK(cudaMalloc((void **)&dOutputProb, sizeof(float) * outputProbSize));
  CUDACHECK(cudaMalloc((void **)&dOutputReg, sizeof(float) * outputRegSize));

  CUDACHECK(cudaMemcpyAsync((void *)dImage, (void *)image,
                            sizeof(float) * imageSize, cudaMemcpyHostToDevice,
                            *stream));

  void *buffers[3];
  buffers[inputIndex] = dImage;
  buffers[outputIndexProb] = dOutputProb;
  buffers[outputIndexReg] = dOutputReg;

  if (!context->enqueue(1, buffers, *stream, nullptr)) {
    std::cout << "Execute failed" << std::endl;
  } else {
    std::cout << "Execute success" << std::endl;
  }

  CUDACHECK(cudaMemcpyAsync((void *)outputProb, (void *)dOutputProb,
                            sizeof(float) * outputProbSize,
                            cudaMemcpyDeviceToHost, *stream));
  CUDACHECK(cudaMemcpyAsync((void *)outputReg, (void *)dOutputReg,
                            sizeof(float) * outputRegSize,
                            cudaMemcpyDeviceToHost, *stream));

  CUDACHECK(cudaStreamSynchronize(*stream));
}
