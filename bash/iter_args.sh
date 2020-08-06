#!/bin/bash
set -eu

total=1

# NB [ -n "$a" ] is true if length of string is non zero
while [ -n "${1:-}" ]; do
    echo "Argument $total is $1"
    total=$(($total+1))
    shift
done
