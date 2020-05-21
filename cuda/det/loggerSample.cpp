#include "logger.h"

int main(int argc, char **argv) {
  gLogger.log(nvinfer1::ILogger::Severity::kINFO, "success");
}
