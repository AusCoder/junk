#!/bin/bash
set -eu

HOST=192.168.1.6

gst-launch-1.0 -v \
               tcpclientsrc host=$HOST port=7469 ! \
               fakesink async=false dump=true
               # rtph265depay ! \
               # h265parse ! \
               # avdec_h265 ! \
               # videoconvert ! \
               # autovideosink
