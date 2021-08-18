#!/usr/bin/env bash
set -eu

FILEPREFIX=gstcustomgl

echo "Compiling $FILEPREFIX"

PKG_LIBS="gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-gl-1.0"

gcc -g -Wall -fPIC \
    $(pkg-config --cflags "$PKG_LIBS") \
    -c -o "$FILEPREFIX.o" "$FILEPREFIX.c"

gcc -shared \
    -o "$FILEPREFIX.so" "$FILEPREFIX.o" \
    $(pkg-config --libs "$PKG_LIBS")
