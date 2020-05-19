#include "trtNet.h"

TrtNet::TrtNet() {
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
  auto parser = nvuffparser::createUffParser();

  // builder->
}
