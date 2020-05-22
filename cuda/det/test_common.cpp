#define CATCH_CONFIG_MAIN

#include "common.h"

#include "catch.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

TEST_CASE("Channel swap - CHWtoHWC - 1x1", "[channelSwap]") {
  float image[3] = {1.0f, 2.0f, 3.0f};
  float out[3];
  channelSwap(image, 1, 1, 3, ChannelSwapType::CHWtoHWC, out);
  REQUIRE(out[0] == 1.0f);
  REQUIRE(out[1] == 2.0f);
  REQUIRE(out[2] == 3.0f);
}

TEST_CASE("Channel swap - CHWtoHWC - 2x3", "[channelSwap]") {
  float image[18] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                     7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                     13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
  float expected[18] = {1.0f, 7.0f,  13.0f, 2.0f, 8.0f,  14.0f,
                        3.0f, 9.0f,  15.0f, 4.0f, 10.0f, 16.0f,
                        5.0f, 11.0f, 17.0f, 6.0f, 12.0f, 18.0f};

  float out[18];
  channelSwap(image, 2, 3, 3, ChannelSwapType::CHWtoHWC, out);

  for (int i = 0; i < 18; i++) {
    REQUIRE(out[i] == expected[i]);
  }
}

TEST_CASE("Channel swap - HWCtoCHW - 1x1", "[channelSwap]") {
  float image[3] = {1.0f, 2.0f, 3.0f};
  float out[3];
  channelSwap(image, 1, 1, 3, ChannelSwapType::HWCtoCHW, out);
  REQUIRE(out[0] == 1.0f);
  REQUIRE(out[1] == 2.0f);
  REQUIRE(out[2] == 3.0f);
}

TEST_CASE("Channel swap - HWCtoCHW - 2x3", "[channelSwap]") {
  float image[18] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                     7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                     13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
  float expected[18] = {1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f,
                        2.0f, 5.0f, 8.0f, 11.0f, 14.0f, 17.0f,
                        3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  float out[18];
  channelSwap(image, 2, 3, 3, ChannelSwapType::HWCtoCHW, out);

  for (int i = 0; i < 18; i++) {
    REQUIRE(out[i] == expected[i]);
  }
}

TEST_CASE("Channel swap idempotent", "[channelSwap]") {
  std::string imagePath = "../../tests/data/execs.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);

  int imageWidth = image.cols;
  int imageHeight = image.rows;
  int depth = image.channels();
  int imageSize = imageHeight * imageWidth * depth;

  std::vector<float> imageArr(imageSize);
  std::vector<float> imageArrT(imageSize);
  std::vector<float> imageArrTT(imageSize);

  for (int i = 0; i < imageHeight; i++) {
    float *ptr = image.ptr<float>(i);
    for (int j = 0; j < imageWidth; j++) {
      imageArr[i * imageWidth + j] = ptr[j];
    }
  }

  channelSwap(imageArr.data(), imageWidth, imageHeight, depth,
              ChannelSwapType::HWCtoCHW, imageArrT.data());
  channelSwap(imageArrT.data(), imageWidth, imageHeight, depth,
              ChannelSwapType::CHWtoHWC, imageArrTT.data());

  for (int i = 0; i < imageSize; i++) {
    REQUIRE(imageArr[i] == imageArrTT[i]);
  }
}
