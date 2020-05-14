#!/bin/bash
set -eu

IMAGE_TAG=junk

# Clean old build files
# find . -name Makefile \
#   | while read x; do; cd $(dirname $x); make clean || true; cd -; done

# Run docker
docker run --rm -ti --gpus=all \
  -e "TERM=xterm-256color" \
  -v $(pwd)/code/junk \
  ${IMAGE_TAG} \
  /bin/zsh
