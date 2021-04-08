#!/bin/bash
set -e

for phase in val test train
do
    # remove old data
    rm -r fashion_data/${phase}K || true
    mkdir fashion_data/${phase}K
    rm -r fashion_data/${phase}SPL2 || true
    mkdir fashion_data/${phase}SPL2
    # compute poses
    echo "Computing poses for $phase:"
    python tool/compute_coordinates.py fashion_data/$phase/ fashion_data/fasion-resize-annotation-${phase}.csv
    # generate pose maps
    echo "Generating pose maps for $phase:"
    python tool/generate_pose_map.py fashion_data/fasion-resize-annotation-${phase}.csv fashion_data/${phase}K/
    # generate parsings
    echo "Generating segmentation maps for $phase:"
    python models/parsing.py fashion_data/${phase}/ fashion_data/${phase}SPL2/
done