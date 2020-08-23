#!/bin/bash
set -eu


    # queue leaky=downstream max-size-time=2000000000 ! \

gst-launch-1.0 -e -v \
               v4l2src device=/dev/video0 ! \
               videoscale ! \
               video/x-raw,height=853,width=480 ! \
               videoconvert ! \
               x264enc ! \
               mp4mux ! \
               filesink location="test.mp4"
               # rtph265pay ! \
               # tcpserversink host=0.0.0.0 port=7469
