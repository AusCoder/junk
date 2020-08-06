#!/bin/bash
set -eu

a_value=
b_value=

while [ -n "${1:-}" ]; do
    case "$1" in
        -a)
            a_value="$2"
            echo "-a passed with value $a_value"
            shift;;
        -b)
            b_value="$2"
            echo "-b passed with value $b_value"
            shift;;
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
