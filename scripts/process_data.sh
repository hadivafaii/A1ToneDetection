#!/bin/bash

nb_std=${1:-1}
base_dir=${2:-"Documents/A1"}

# if [[ $base_dir == "null" ]]; then
#   if [[ $(uname -n) == "V1" ]]; then
#     base_dir="Documents/Kanold"
#   elif [[ $(uname -n) == "SigurRos" ]]; then
#     base_dir="Documents/PROJECTS/Kanold"
#   fi
# fi

cd ..

python3 -m utils.process --nb_std $nb_std --base_dir $base_dir --verbose

echo Done!
