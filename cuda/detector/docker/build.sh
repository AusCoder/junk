#!/bin/bash
set -eu

if [ ! -f docker/Dockerfile ]
then
  echo Could not find ./docker/Dockerfile
  exit 1
fi

CUDA_VERSION=10.2
CUDNN_VERSION=7.6
TENSORRT_VERSION=7.0.0.11

mkdir -p docker/tmp

CUDA_DIR=docker/tmp/cuda-${CUDA_VERSION}
TENOSRRT_DIR=docker/tmp/TensorRT-${TENSORRT_VERSION}-cuda-${CUDA_VERSION}-cudnn-${CUDNN_VERSION}

if [ ! -d ${CUDA_DIR} ]
then
  echo Preparing Cuda
  sh /data/cuda/packages/cuda_10.2.89_440.33.01_linux.run --target ${CUDA_DIR}-tmp --noexec
  mv ${CUDA_DIR}-tmp/builds/cuda-toolkit ${CUDA_DIR}
  rm -rf ${CUDA_DIR}-tmp
fi

if [ ! -f ${CUDA_DIR}/include/cudnn.h ]
then
  echo Preparing Cudnn
  mkdir -p docker/tmp/cudnn-tmp
  tar xvf /data/cuda/packages/cudnn-10.2-linux-x64-v7.6.5.32.tgz --directory docker/tmp/cudnn-tmp
  mv docker/tmp/cudnn-tmp/cuda/include/* ${CUDA_DIR}/include
  mv docker/tmp/cudnn-tmp/cuda/lib64/* ${CUDA_DIR}/lib64
  rm -rf docker/tmp/cudnn-tmp
fi

if [ ! -d ${TENOSRRT_DIR} ]
then
  echo Preparing TensorRT
  mkdir -p ${TENOSRRT_DIR}-tmp
  tar xvf /data/cuda/packages/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz --directory ${TENOSRRT_DIR}-tmp
  mv ${TENOSRRT_DIR}-tmp/TensorRT-7.0.0.11 ${TENOSRRT_DIR}
  rmdir ${TENOSRRT_DIR}-tmp
fi

export DOCKER_BUILDKIT=1
docker build \
  -t junk/cuda/detector \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
  --build-arg TENSORRT_VERSION=${TENSORRT_VERSION} \
  --build-arg CUDA_DIR=${CUDA_DIR} \
  --build-arg TENOSRRT_DIR=${TENOSRRT_DIR} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  .
