#!/bin/bash

# Setup script for image generation model benchmarking

# Directories for images
coco_30k_image_dir="/home/lixiang/data/fid_kolors_torch"

# CSV file path
csv_path="resources/MS-COCO_val2014_30k_captions.csv"

# Log file path
log_file="benchmark_log.txt"

# Redirect all output and errors to log file
exec > >(tee -a "$log_file") 2>&1

echo "Starting the benchmarking process at $(date)"

# Check if image directory exists
if [ ! -d "$coco_30k_image_dir" ]; then
    echo "Error: Image directory does not exist: $coco_30k_image_dir"
    exit 1
fi

# Check if CSV file exists
if [ ! -f "$csv_path" ]; then
    echo "Error: CSV file does not exist: $csv_path"
    exit 1
fi

# Execute Python scripts for different metrics and log results
echo "Calculating FID Score..."
python3 metrics/fid_score.py "$coco_30k_image_dir"

echo "Calculating Inception Score..."
python3 metrics/inception_score.py --path "$coco_30k_image_dir"

echo "Calculating CLIP Score..."
python3 metrics/clip_score.py "$coco_30k_image_dir" "$csv_path"

echo "Calculating Aesthetic Score..."
python3 metrics/aesthetic_score.py --image_path "$coco_30k_image_dir"

echo "Calculating Structural Similarity Index..."
python3 metrics/structural_similarity.py --folder1 "$coco_30k_image_dir" --folder2 "$coco_30k_image_dir"

echo "Benchmarking process completed at $(date)"
