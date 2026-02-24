#!/bin/bash

# Change the following to your path
SKELETON_ROOT="/PATH/TO/KEYPOINT_DATA"
SILHOUETTE_ROOT="/PATH/TO/SILHOUETTE_DATA"
OUTPUT_ROOT="/PATH/TO/OUTPUT_DIRECTORY"
LOG_DIR="/PATH/TO/LOG_FILE/transform.log"
SCORE_THRESHOLD=0.3

python transform.py \
    --skeleton_root "$SKELETON_ROOT" \
    --silhouette_root "$SILHOUETTE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --log_dir "$LOG_DIR" \
    --score_threshold "$SCORE_THRESHOLD"