#!/bin/bash
set -eu

IMAGE_TAG=junk

export DOCKER_BUILDKIT=1
docker build \
  -t ${IMAGE_TAG} \
  .
