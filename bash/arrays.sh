#!/bin/bash
set -euo pipefail

arr=(1 2 3)
echo echoing all of an array "${arr[@]}"
echo echoing an array as a string "${arr[*]}"
echo echoing an array ${arr}, it just shows the first element

echo

arr=("a" "b c" "d")
echo Now for this challenging array
echo echoing arr with the '${arr[*]}' syntax
for x in ${arr[*]}; do
    echo $x
done

echo echoing arr with the '"${arr[@]}"' syntax
echo NB: we need the quotes when doing with @
for x in "${arr[@]}"; do
    echo $x
done

function enumerate_array () {
    echo Array enumerate function
    # You need to wrap in ( ) to turn input args into an array
    # (Otherwise it is a string of args?)
    local ar=("$@")
    # This gets length of array
    echo Array length ${#ar[@]}
    # This gets indexes of an array
    for i in "${!ar[@]}"; do
        # These do array indexing
        echo $i: ${ar[i]}, ${ar[$i]}, ${ar["$i"]}
    done
}

echo
enumerate_array "${arr[@]}"
echo
enumerate_array 1 2 3

function build_array () {
    local ar=()
    for i in $(seq 1 5); do
        ar+=($((3*$i)))
    done
    echo "${ar[@]}"
}
arr=($(build_array))  # NB: arr=("$(build_array)") does not work
echo
echo Built array length "${#arr[@]}" is "${arr[@]}"
for x in "${arr[@]:1:3}"; do
    echo Sliced array $x
done

echo
lsstr=$(ls)
lsstrquotes="$(ls)"
lsarr=($(ls))
lsarrstr=("$(ls)")
echo ls output as str: $lsstr
echo
echo ls output as quoted str: $lsstrquotes
echo
echo ls output as array: length ${#lsarr[@]} with elements: "${lsarr[@]}"
echo
echo ls output as string in array: length ${#lsarrstr[@]} with elements: "${lsarrstr[@]}"
