#!/bin/bash
set -eu

file=$1
count=1

if false; then
    cat $file | while read line; do
        echo "$count: $line"
        count=$(($count + 1))
    done
fi

while read line; do
    echo "$count: $line"
    count=$(($count + 1))
done < $file
