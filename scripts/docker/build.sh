#!/bin/bash
set -eu

IMAGE_TAG=junk

# NB: you might need to login to get the nvidia images
if false
then
  docker login nvcr.io -u '$oauthtoken' -p $NVIDIA_NGC_API_KEY
fi

export DOCKER_BUILDKIT=1
docker build \
  --progress plain \
  -t ${IMAGE_TAG} \
  .
