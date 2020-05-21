
#include "logger.h"
#include <iostream>

using Severity = nvinfer1::ILogger::Severity;

void Logger::log(Severity severity, const char *msg) {
  std::cout << renderSeverity(severity) << msg << std::endl;
};

std::string Logger::renderSeverity(Severity severity) {
  if (severity == Severity::kERROR) {
    return "[ERROR] ";
  } else if (severity == Severity::kWARNING) {
    return "[WARNING] ";
  } else if (severity == Severity::kINFO) {
    return "[INFO] ";
  } else if (severity == Severity::kVERBOSE) {
    return "[VERBOSE] ";
  } else {
    return "[UNKNOWN] ";
  }
}

Logger gLogger;
