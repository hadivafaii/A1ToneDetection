#!/bin/bash

base_dir=${1:-null}

if [[ $base_dir == "null" ]]; then
  if [[ $(uname -n) == "V1" ]]; then
    base_dir="Documents/Kanold"
  elif [[ $(uname -n) == "SigurRos" ]]; then
    base_dir="Documents/PROJECTS/Kanold"
  fi
fi

cd ..

python3 -m analysis.lda_analysis --base_dir $base_dir --verbose

echo Done!
