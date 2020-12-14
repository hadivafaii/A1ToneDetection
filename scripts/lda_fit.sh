#!/bin/bash

base_dir=${1:-"Documents/A1"}


cd ..

python3 -m analysis.lda_analysis --base_dir $base_dir --verbose

echo Done!
