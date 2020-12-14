#!/bin/bash

cm=$1
clf_type=${2:-logreg}
fps=${3:-2}
dpi=${4:-100}
base_dir=${5:-"Documents/A1"}

cd ..

python3 -m utils.animation $cm --clf_type $clf_type --fps $fps --dpi $dpi --base_dir $base_dir --normalize --verbose

echo Done!
