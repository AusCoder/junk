FROM ubuntu:18.04

ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG TENSORRT_VERSION

ARG CUDA_DIR
ARG TENOSRRT_DIR

COPY ${CUDA_DIR} /opt/cuda-${CUDA_VERSION}
COPY ${TENOSRRT_DIR} /opt/TensorRT-${TENSORRT_VERSION}-cuda-${CUDA_VERSION}-cudnn-${CUDNN_VERSION}

# RUN \
#   mkdir -p /opt/cuda && \
#   mkdir -p /opt/cuda/cuda-${CUDA_VERSION}
# RUN sh /opt/cuda/packages/cuda-${CUDA_VERSION}.run --target /opt/cuda/cuda-${CUDA_VERSION} --noexec
