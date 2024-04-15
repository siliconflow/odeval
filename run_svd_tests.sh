#!/bin/bash

# Setup script for image generation model benchmarking

# For CN: Set environment variables for Hugging Face endpoint
export HF_ENDPOINT="https://hf-mirror.com"

REPO_ID="siliconflow/stable-video-diffusion-img2vid-xt-deepcache-int8"

MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$REPO_ID'))")
if [ $? -ne 0 ]; then
    echo "Model download failed"
    exit 1
fi

OUTPUT_DIR="/path/to/your/output"
PROMPTS_DIR="/path/to/prompts"

python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --model $MODEL_PATH --deepcache
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --deepcache
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --compiler none
  
evaluate_clip_score() {
    local category="$1"
    python -m clip_score "$OUTPUT_DIR/$category" "$PROMPTS_DIR/$category"
}

for category in anime concept-art paintings photo; do
    evaluate_clip_score "$category"
done

python3 aesthetic_score.py --image_path $OUTPUT_DIR

python3 inception_score.py --path $OUTPUT_DIR
