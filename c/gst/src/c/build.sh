#!/usr/bin/env bash
set -eu

FILEPREFIX=gstcustomgl

echo "Compiling $FILEPREFIX"

PKG_LIBS="gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-gl-1.0 opencv4"
CUDA_ROOT_DIR="/data/cuda/cuda-11.0-cudnn-8005"
CUDA_INC_DIR="$CUDA_ROOT_DIR/include"
CUDA_LIB_DIR="$CUDA_ROOT_DIR/lib64"

gcc -g -Wall -fPIC \
    $(pkg-config --cflags "$PKG_LIBS") \
    "-I$CUDA_INC_DIR" \
    -c -o "$FILEPREFIX.o" "$FILEPREFIX.cc"

gcc -shared \
    -o "$FILEPREFIX.so" "$FILEPREFIX.o" \
    $(pkg-config --libs "$PKG_LIBS") \
    "-L$CUDA_LIB_DIR" -lcudart
