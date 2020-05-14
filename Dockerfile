# TensorRT base
FROM nvcr.io/nvidia/tensorrt:20.03-py3

ARG OPENCV_VERSION=4.3.0

# Cuda base and then install TensorRT?
# FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN \
  apt update && \
  apt upgrade -y
RUN \
  apt install -y --no-install-recommends \
    zsh \
    build-essential cmake unzip pkg-config wget \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    python3-dev python3-numpy

RUN \
  apt install -y --no-install-recommends \
    bash

# Build opencv with cuda support
RUN \
  mkdir -p /opt/opencv/src && \
  cd /opt/opencv/src && \
  wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz && \
  mkdir -p opencv && \
  tar xzf opencv.tar.gz --directory ./opencv && \
  rm -f opencv.tar.gz

RUN \
  cd /opt/opencv/src && \
  wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz && \
  mkdir -p opencv_contrib && \
  tar xzf opencv_contrib.tar.gz --directory ./opencv_contrib && \
  rm -f opencv_contrib.tar.gz

RUN \
  cd /opt/opencv/src/opencv/opencv-${OPENCV_VERSION} && \
  mkdir -p build && \
  cd build && \
  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/opencv/install \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/src/opencv_contrib/opencv_contrib-4.3.0/modules \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON_INCLUDE_DIR=/usr/include/python3.6m \
    -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.6/dist-packages/numpy/core/include \
    ..

RUN \
  cd /opt/opencv/src/opencv/opencv-${OPENCV_VERSION}/build && \
  make -j 8 && \
  make install

# ENV CUDA_INSTALL_DIR=
# ENV OPENCV_INCLUDE_DIR=/opt/opencv/install/include
# ENV LD_LIBRARY_PATH=/opt/opencv/install/lib:$LD_LIBRARY_PATH
