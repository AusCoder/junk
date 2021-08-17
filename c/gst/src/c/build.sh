#!/usr/bin/env bash
set -eu

FILEPREFIX=gstcustomelem

gcc -g -Wall -fPIC \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0) \
    -c -o "$FILEPREFIX.o" "$FILEPREFIX.c"

gcc -shared \
    -o "$FILEPREFIX.so" "$FILEPREFIX.o" \
    $(pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
