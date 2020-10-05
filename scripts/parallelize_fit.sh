#!/bin/bash

cm=$1
clf_type=$2
penalty=${3:-l1}

echo "cm: $cm"
echo "clf_type: $clf_type"
echo "penalty: $penalty"

cd ..

C_ARR=( 100.0 10.0 5.0 1.0 0.07 0.05 0.03 0.02 0.01 0.007 0.005 0.003 0.001 0.0005 0.0001 )


mk_screens () {
  local -n arr=$1
  for c in "${arr[@]}"; do
    screen -dmS "C_${c}"
    echo $c
  done
}

fit () {
  local -n arr=$4
  for c in "${arr[@]}"; do
    screen -S "C_${c}" -X stuff "python3 -m analysis.analysis $1 --clf_type $2 --penalty $3 -C $c --machine $(uname -n) --verbose ^M"
  done
}

mk_screens C_ARR
fit $cm $clf_type $penalty C_ARR

echo Done!
