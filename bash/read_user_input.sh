#!/bin/bash
set -eu

if false; then
    echo -n "What's your name? "
    read name
    echo "Your name is $name"
fi

if false; then
    read -p "What's your name? " first last  # You can use 2 vars
    echo "Your first name is $first and last name is $last"
fi

if true; then
    read -s -p "Your password: " password
    echo
    echo "Your password is $password"
fi
