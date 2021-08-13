#!/usr/bin/env bash
set -eu

gcc -g -Wall -fPIC \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0) \
    -c -o gstmyfirstelement.o gstmyfirstelement.c

gcc -shared \
    -o gstmyfirstelement.so gstmyfirstelement.o \
    $(pkg-config --libs gstreamer-1.0 gstreamer-base-1.0)
