#!/bin/bash
set -eu

gst-launch-1.0 -e \
  v4l2src device=/dev/video0 ! \
  videoconvert ! \
  x264enc ! \
  mp4mux ! \
  filesink location="test.mp4"
