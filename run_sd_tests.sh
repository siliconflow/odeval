#!/bin/bash

# Setup script for image generation model benchmarking

# For CN: Set environment variables for Hugging Face endpoint
export HF_ENDPOINT="https://hf-mirror.com"

REPO_ID="siliconflow/stable-diffusion-v1-5-onediff-enterprise"

MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$REPO_ID'))")
if [ $? -ne 0 ]; then
    echo "Model download failed"
    exit 1
fi

REPO_DEEPCACHE_ID="siliconflow/stable-diffusion-v1-5-onediff-deepcache-int8"

MODEL_DEEPCACHE_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$REPO_DEEPCACHE_ID'))")
if [ $? -ne 0 ]; then
    echo "Model download failed"
    exit 1
fi

OUTPUT_DIR="/path/to/your/output"
PROMPTS_DIR="/path/to/prompts"

python3 sd/text_to_image_sd1_5_enterprise_quality_benchmark.py --model $MODEL_DEEPCACHE_PATH --image_path $OUTPUT_DIR --deep_cache True
python3 sd/text_to_image_sd1_5_enterprise_quality_benchmark.py --model $MODEL_PATH --image_path $OUTPUT_DIR 
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR 
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR --deep_cache False
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR --compile False --deep_cache False

evaluate_clip_score() {
    local category="$1"
    python -m clip_score "$OUTPUT_DIR/$category" "$PROMPTS_DIR/$category"
}

for category in anime concept-art paintings photo; do
    evaluate_clip_score "$category"
done

python3 aesthetic_score.py --image_path $OUTPUT_DIR

python3 inception_score.py --path $OUTPUT_DIR
