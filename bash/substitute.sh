#!/bin/bash
set -eu

# Lessons from:
#    https://github.com/anordal/shellharden/blob/master/how_to_do_things_safely_in_bash.md

# Use quotes for variable substitution
cmd="import math; print(f'the value is {math.exp(1)}')"

python -c "$cmd"

value="$(python -c "$cmd")"
echo $value

FILE_LOCATION=".tmp file with a space sucker - this is why you need to quote variables and cmd substitutions!.txt"
rm -f "$FILE_LOCATION"
for i in seq 1 3; do
    echo abc >> "$FILE_LOCATION"
done
gst-launch-1.0 filesrc location="$FILE_LOCATION" ! fakesink sync=false async=false dump=true

# Use arrays
arr=(
    a
    b
    c
)
for a in "${arr[@]}"; do
    echo "$a"
done

args=(
    -c
    "import math; print(f\"sin pi/2 is roughly: {math.sin(1.71)}\")"
)
python "${args[@]}"

# Splitting a string on a separator
bad_str="this is bad string"
bad_arr=($bad_str)
echo bad arr len ${#bad_arr[@]}

good_str="this is good string"
good_arr=()
while read -rd " " i; do
    good_arr+=("$i")
done < <(printf "%s%s" "$good_str" " ")
echo good arr len "${#good_arr[@]}" with value "${good_arr[@]}"
