#!/bin/bash

cm=$1
clf_type=${2:-svm}
base_dir=${3:-"Documents/V1"}

cd ..

python3 -m analysis.summarize_results $cm --clf_type $clf_type --base_dir $base_dir --verbose

echo Done!
