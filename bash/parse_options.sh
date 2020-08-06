#!/bin/bash
set -eu

while [ -n "${1:-}" ]; do
    case "$1" in
        -a) echo "-a option passed";;
        -b) echo "-b option passed";;
        -c) echo "-c option passed";;
        *) echo "Option $1 not recognized";;
    esac
    shift
done
