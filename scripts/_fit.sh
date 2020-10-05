#!/bin/bash

cm=$1
clf_type=$2
penalty=$3

fit () {
  C_ARRAY=( 100.0 10.0 5.0 1.0 0.07 0.05 0.03 0.02 0.01 0.007 0.005 0.003 0.001 0.0005 0.0001 )
  for c in ${C_ARRAY[@]}; do
    sem -j +0 python3 ../analysis/analysis.py $1 --clf_type $2 --penalty $3 -C $c --machine $(uname -n)
  done
}

fit $cm $clf_type $penalty
