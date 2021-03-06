#ifndef _LOGGER_H
#define _LOGGER_H

#include "NvInferRuntimeCommon.h"
#include <string>

class Logger : public nvinfer1::ILogger {
public:
  void log(nvinfer1::ILogger::Severity severity, const char *msg) override;

private:
  static std::string renderSeverity(nvinfer1::ILogger::Severity severity);
};

extern Logger gLogger;

#endif // _LOGGER_H
