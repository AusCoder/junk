#!/bin/bash
set -eu

while [ -n "${1:-}" ]; do
    case "$1" in
        -a) echo "-a option passed";;
        -b) echo "-b option passed";;
        -c) echo "-c option passed";;
        --)  # At double dash, stop parsing options and treat rest as args
            shift
            break;;
        *) echo "Option $1 not recognized";;
    esac
    shift
done


total=1
while [ -n "${1:-}" ]; do
    echo "Argument $total is $1"
    total=$(($total+1))
    shift
done
