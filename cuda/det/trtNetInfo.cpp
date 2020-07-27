#include "trtNetInfo.h"
#include "commonCuda.h"

#include <fstream>
#include <json/json.h>
#include <sstream>
#include <stdexcept>

TensorInfo::TensorInfo(std::string n, nvinfer1::Dims3 s, TensorInputOrder i)
    : name{n}, shape{s}, inputOrder{i} {}

TensorInfo::TensorInfo(std::string n, nvinfer1::Dims3 s)
    : name{n}, shape{s}, inputOrder{TensorInputOrder::None} {}

std::string TensorInfo::render() const {
  std::stringstream ss;
  std::string orderStr;
  if (inputOrder == TensorInputOrder::NHWC) {
    orderStr = "NHWC";
  } else {
    orderStr = "None";
  }
  ss << "TensorInfo(name=" << name << ", shape=(" << shape.d[0] << ", "
     << shape.d[1] << ", " << shape.d[2] << ")"
     << ", order=" << orderStr << ")";
  return ss.str();
}

int TensorInfo::getHeight() const { return shape.d[0]; }
int TensorInfo::getWidth() const { return shape.d[1]; }
int TensorInfo::volume() const { return dimsSize(shape); }

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

static nvinfer1::Dims3 jsonGetShape(const Json::Value &v,
                                    const std::string &key) {
  std::vector<int> ds;
  if (v.isMember(key) && v[key].isArray()) {
    for (auto &v : v[key]) {
      if (!v.isNull()) {
        ds.push_back(v.asInt());
      }
    }
    if (ds.size() != 3) {
      throw std::invalid_argument("Unexpected number of dims in shape");
    }
  } else {
    throw std::invalid_argument(key + " key not valid");
  }
  return nvinfer1::Dims3{ds[0], ds[1], ds[2]};
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
        TensorInputOrder inputOrder;
        std::string name;
        nvinfer1::Dims3 dims;

        if (jsonGetString(v, "format") == "NHWC") {
          inputOrder = TensorInputOrder::NHWC;
        } else {
          inputOrder = TensorInputOrder::None;
        }
        name = jsonGetString(v, "name");
        dims = jsonGetShape(v, "shape");
        netInfo.inputTensorInfos.push_back({name, dims, inputOrder});
      }
    } else {
      throw std::invalid_argument("inputs key not valid");
    }

    if (value.isMember("outputs")) {
      for (auto &v : value["outputs"]) {
        std::string name;
        nvinfer1::Dims3 dims;

        name = jsonGetString(v, "name");
        dims = jsonGetShape(v, "shape");
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
