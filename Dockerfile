# TensorRT base
FROM nvcr.io/nvidia/tensorrt:20.03-py3

ARG OPENCV_VERSION=4.3.0

# Cuda base and then install TensorRT?
# FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN
  apt update && \
  apt upgrade
RUN \
  apt install \
    zsh \
    build-essential cmake unzip pkg-config wget \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    python3-dev python3-numpy

# Build opencv with cuda support
RUN \
  mkdir -p /opt/opencv/src && \
  cd /opt/opencv/src && \
  \
  wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz && \
  mkdir -p opencv && \
  tar xzf opencv.tar.gz --directory ./opencv && \
  rm -f opencv.tar.gz && \
  \
  wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
  mkdir -p opencv_contrib && \
  tar xzf opencv_contrib.tar.gz --directory opencv_contrib && \
  rm -f opencv_contrib.tar.gz

# RUN \
#   cd /opt/opencv/src/opencv-${OPENCV_VERSION} && \
#   mkdir -p build && \
#   cd build && \
#   cmake \
#     -D CMAKE_BUILD_TYPE=RELEASE \
#     -D WITH_CUDA=ON \
#     -D CUDA_____DIR= \
#     -D CMAKE_INSTALL_PREFIX=/opt/opencv/install \
#     -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/src/opencv_contrib/modules \
#     -D PYTHON_EXECUTABLE=/usr/bin/python3 \
#     ..

# RUN make -j 8 && make install
