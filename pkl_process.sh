#!/bin/bash

# Change the following to your path
SIL_PATH="/PATH/TO/TRANSFORMED_DATA"
OUTPUT_PATH="/PATH/TO/OUTPUT_DIRECTORY"

python datasets/Gait3D/process_gait3d_alignment.py \
    --sil_path "$SIL_PATH" \
    --output_path "$OUTPUT_PATH"