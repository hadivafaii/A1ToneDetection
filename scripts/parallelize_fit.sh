#!/bin/bash

cm=$1
clf_type=$2
penalty=${3:-l1}

echo "cm: $cm"
echo "clf_type: $clf_type"
echo "penalty: $penalty"

cd ..

C_ARR=( 1.5 1.0 0.07 0.05 0.03 0.02 0.01 0.007 0.005 0.003 0.001 0.0005 0.0001 0.00005 0.00001 )


mk_screens () {
  local -n arr=$1
  for c in "${arr[@]}"; do
    screen -dmS "C_${c}"
    echo "screen created: C_$c"
  done
}

fit () {
  available_mem=$(awk '/MemFree/ { printf "%d \n", $2/1024/1024 }' /proc/meminfo)
  min_mem=60
  c_border=0.005

  local -n arr=$4
  for c in "${arr[@]}"; do
    if (($available_mem < $min_mem)); then
      screen -S "C_${c}" -X stuff "python3 -m analysis.analysis $1 --clf_type $2 --penalty $3 -C $c --machine $(uname -n) --verbose --save_to_pieces ^M"
      # this will prevent Kill due to lack of memory
      if (( $(echo "$c == $c_border" |bc -l) )); then
        sleep 3h
      fi
    else
      screen -S "C_${c}" -X stuff "python3 -m analysis.analysis $1 --clf_type $2 --penalty $3 -C $c --machine $(uname -n) --verbose ^M"
    fi
  done
}

mk_screens C_ARR
fit $cm $clf_type $penalty C_ARR

echo Done!
