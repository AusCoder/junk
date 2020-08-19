#include "trtNetInfo.h"
#include "commonCuda.hpp"

#include <algorithm>
#include <fstream>
#include <json/json.h>
#include <numeric>
#include <sstream>
#include <stdexcept>

TensorInfo::TensorInfo(std::string n, std::vector<int> s, TensorInputOrder i)
    : name{n}, shape{s}, inputOrder{i} {}

TensorInfo::TensorInfo(std::string n, std::vector<int> s)
    : name{n}, shape{s}, inputOrder{TensorInputOrder::None} {}

std::string TensorInfo::render() const {
  std::stringstream ss;
  std::string orderStr;
  if (inputOrder == TensorInputOrder::NHWC) {
    orderStr = "NHWC";
  } else {
    orderStr = "None";
  }
  ss << "TensorInfo(name=" << name << ", shape=(" << shape.at(0) << ", "
     << shape.at(1) << ", " << shape.at(2) << ")"
     << ", order=" << orderStr << ")";
  return ss.str();
}

int TensorInfo::getHeight() const { return shape.at(0); }
int TensorInfo::getWidth() const { return shape.at(1); }
int TensorInfo::volume() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}
nvinfer1::Dims3 TensorInfo::getInputDims() const {
  return {shape.at(0), shape.at(1), shape.at(2)};
}

std::string TrtNetInfo::render() const {
  std::stringstream ss;
  ss << "TrtNetInfo(\n  inputTensorInfos=[\n";
  for (auto &x : inputTensorInfos) {
    ss << "    " << x.render() << "\n";
  }
  ss << "  ],\n  outputTensorInfos=[\n";
  for (auto &x : outputTensorInfos) {
    ss << "    " << x.render() << "\n";
  }
  ss << "  ]\n)";
  return ss.str();
}

static std::string jsonGetString(const Json::Value &v, const std::string &key) {
  if (v.isMember(key) && v[key].isString()) {
    return v[key].asString();
  } else {
    throw std::invalid_argument(key + " key not valid");
  }
}

static std::vector<int> jsonGetShape(const Json::Value &v,
                                     const std::string &key) {
  std::vector<int> ds;
  if (v.isMember(key) && v[key].isArray()) {
    for (auto &v : v[key]) {
      if (!v.isNull()) {
        ds.push_back(v.asInt());
      }
    }
    if (!(ds.size() == 1 or ds.size() == 3)) {
      throw std::invalid_argument("Unexpected number of dims in shape");
    }
  } else {
    throw std::invalid_argument(key + " key not valid");
  }
  return ds;
}

TrtNetInfo TrtNetInfo::readTrtNetInfo(const std::string &netInfoPath) {
  TrtNetInfo netInfo{};
  std::ifstream netInfoFile{netInfoPath};
  if (netInfoFile) {
    Json::Reader reader{};
    Json::Value value{};
    reader.parse(netInfoFile, value);
    if (value.isMember("inputs") && value["inputs"].isArray()) {
      for (auto &v : value["inputs"]) {
        TensorInputOrder inputOrder{TensorInputOrder::None};
        if (jsonGetString(v, "format") == "NHWC") {
          inputOrder = TensorInputOrder::NHWC;
        }
        auto name = jsonGetString(v, "name");
        auto dims = jsonGetShape(v, "shape");
        netInfo.inputTensorInfos.push_back({name, dims, inputOrder});
      }
    } else {
      throw std::invalid_argument("inputs key not valid");
    }

    if (value.isMember("outputs")) {
      for (auto &v : value["outputs"]) {
        auto name = jsonGetString(v, "name");
        auto dims = jsonGetShape(v, "shape");
        netInfo.outputTensorInfos.push_back({name, dims});
      }
    } else {
      throw std::invalid_argument("No outputs key");
    }
  } else {
    // TODO: change error type
    throw std::invalid_argument("Couldn't read net description file: " +
                                netInfoPath);
  }
  return netInfo;
}
