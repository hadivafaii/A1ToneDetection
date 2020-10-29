#!/bin/bash

cm=$1
clf_type=${2:-svm}
penalty=${3:-l1}
base_dir=${4:-null}

if [[ $base_dir == "null" ]]; then
  if [[ $(uname -n) == "V1" ]]; then
    base_dir="Documents/Kanold"
  elif [[ $(uname -n) == "SigurRos" ]]; then
    base_dir="Documents/PROJECTS/Kanold"
  fi
fi

echo "cm: $cm"
echo "clf_type: $clf_type"
echo "penalty: $penalty"
echo "base_dir: $base_dir"

cd ..

# C_ARR=( 0.001 0.005 0.01 0.05 0.1 0.5 1.0 )
C_ARR=( 0.0001 0.000001 )

mk_screens () {
  local -n arr=$1
  for c in "${arr[@]}"; do
    screen -dmS "${2}_C=${c}_cm=${3}"
    echo "screen created: ${2}_C=${c}_cm=${3}"
  done
}

fit () {
  local -n arr=$5
  for c in "${arr[@]}"; do
    screen -S "${2}_C=${c}_cm=${1}" -X stuff "python3 -m analysis.clf_analysis $1 --clf_type $2 --penalty $3 --base_dir $4 -C $c --hidden_size $1 --verbose --save_to_pieces ^M"
  done
}

mk_screens C_ARR $clf_type $cm
fit $cm $clf_type $penalty $base_dir C_ARR

echo Done!

# for the case of mlp classifier, $cm is used interchangibly with hidden_size
