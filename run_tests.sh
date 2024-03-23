#!/bin/bash

MODEL_PATH="/path/to/model"
OUTPUT_DIR="/path/to/output"

declare -a PROMPTS=("prompt1" "prompt2" "prompt3")

python generate_images.py --model_path $MODEL_PATH --output_dir $OUTPUT_DIR --prompts "${PROMPTS[@]}"
