#!/bin/bash

cm=$1
clf_type=$2
penalty=${3:-l1}

echo "cm: $cm"
echo "clf_type: $clf_type"
echo "penalty: $penalty"

cd ..

C_ARR=( 0.001 0.005 0.01 0.05 0.1 0.5 1.0 )


mk_screens () {
  local -n arr=$1
  for c in "${arr[@]}"; do
    screen -dmS "C_${2}_${c}"
    echo "screen created: C_${2}_${c}"
  done
}

fit () {
  local -n arr=$4
  for c in "${arr[@]}"; do
    screen -S "C_${2}_${c}" -X stuff "python3 -m analysis.clf_analysis $1 --clf_type $2 --penalty $3 -C $c --machine $(uname -n) --verbose --save_to_pieces ^M"
  done
}

mk_screens C_ARR $clf_type
fit $cm $clf_type $penalty C_ARR

echo Done!
