#!/bin/bash

# Setup script for image generation model benchmarking with selectable dataset

# Default dataset is 'coco'
dataset="${1:-coco}"

if [ "$dataset" == "coco" ]; then
    output_image_dir_torch_ref="/path/to/kolors_torch_${dataset}"
    # Different acceleration backends recommend using different folders.
    output_image_dir="/path/to/kolors_oneflow_${dataset}" # or output_image_dir="/path/to/kolors_nexfort_${dataset}"
    csv_path="resources/MS-COCO_val2014_30k_captions.csv"
elif [ "$dataset" == "hps" ]; then
    output_image_dir="/path/to/kolors_nexfort_${dataset}"
else
    echo "Unsupported dataset: $dataset"
    exit 1
fi

# Log file path
log_file="benchmark_log_${dataset}.log"

exec > >(tee -a "$log_file") 2>&1

echo "Starting the benchmarking process for $dataset at $(date)"

if [ ! -d "$output_image_dir" ]; then
    echo "Error: Image directory does not exist: $output_image_dir"
    exit 1
fi


if [ "$dataset" == "coco" ]; then
    echo "Calculating FID Score..."
    python3 metrics/fid_score.py "$output_image_dir"

    echo "Calculating Inception Score..."
    python3 metrics/inception_score.py --path "$output_image_dir"

    echo "Calculating CLIP Score..."
    python3 metrics/clip_score.py "$output_image_dir" "$csv_path"

    echo "Calculating Aesthetic Score..."
    python3 metrics/aesthetic_score.py --image_path "$output_image_dir"

    echo "Calculating Structural Similarity Index..."
    python3 metrics/structural_similarity.py --folder1 "$output_image_dir_torch_ref" --folder2 "$output_image_dir"
elif [ "$dataset" == "hps" ]; then
    echo "Calculating HPS v2 Score..."
    python3 metrics/human_preference_score.py --image_path "$output_image_dir"

fi

echo "Benchmarking process completed for $dataset at $(date)"
