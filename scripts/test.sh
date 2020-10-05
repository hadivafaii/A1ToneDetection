#!/bin/bash

while IFS= read -r line; do
    echo "Text read from file: $line"
done < my_filename.txt


cd ../analysis
$search_dir = ./

for entry in "$search_dir"/*
do
  echo "$entry"
done
