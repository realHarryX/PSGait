#!/bin/bash

python parsing.py \
    --silhouette_root /PATH/TO/SILHOUETTE_DATA \
    --keypoint_root /PATH/TO/KEYPOINT_DATA \
    --output_root /PATH/TO/OUTPUT_DIRECTORY \
    --log_file /PATH/TO/LOG_FILE/parsing.log \
    --conf_threshold 0.1 \
    --circle_r 16 \
    --line_width 24 \
    --num_processes 36