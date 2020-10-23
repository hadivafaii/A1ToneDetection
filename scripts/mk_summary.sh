#!/bin/bash

cm=$1
clf_type=${2:-svm}
base_dir=${3:-null}

if [[ $base_dir == "null" ]]; then
  if [[ $(uname -n) == "V1" ]]; then
    base_dir="Documents/Kanold"
  elif [[ $(uname -n) == "SigurRos" ]]; then
    base_dir="Documents/PROJECTS/Kanold"
  fi
fi

cd ..

python3 -m analysis.summarize_results $cm --clf_type $clf_type --base_dir $base_dir --verbose

echo Done!
