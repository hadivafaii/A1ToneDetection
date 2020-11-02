#!/bin/bash

cm=$1
clf_type=${2:-svm}
start_time=${3:-30}
end_time=${4:-75}
filter_sz=${5:-5}
threshold=${6:-0.9}
base_dir=${7:-null}

if [[ $base_dir == "null" ]]; then
  if [[ $(uname -n) == "V1" ]]; then
    base_dir="Documents/Kanold"
  elif [[ $(uname -n) == "SigurRos" ]]; then
    base_dir="Documents/PROJECTS/Kanold"
  fi
fi

cd ..

python3 -m analysis.clf_process $cm --clf_type $clf_type --start_time $start_time --end_time $end_time --filter_sz $filter_sz --threshold $threshold  --base_dir $base_dir --verbose

echo Done!
