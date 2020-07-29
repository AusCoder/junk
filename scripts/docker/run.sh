#!/bin/bash
set -eu

IMAGE_TAG=junk
CONTAINER_NAME=junk

# Clean old build files
# find . -name Makefile \
#   | while read x; do; cd $(dirname $x); make clean || true; cd -; done

if docker ps -a --format "{{.Names}}" | grep -q -E "^$CONTAINER_NAME$"
then
  # Start the existing container if it exists
  docker start -ai junk
else
  # Run docker
  # The security options are for gdb
  docker run \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --name $CONTAINER_NAME \
    -ti \
    --gpus=all \
    --workdir /code/junk/cuda/det \
    -e "TERM=xterm-256color" \
    -v $(pwd):/code/junk \
    ${IMAGE_TAG} \
    /bin/zsh
fi


# Try the following:
# https://github.com/bminor/binutils-gdb
# ./configure --prefix=/opt/gdb
# make && make install