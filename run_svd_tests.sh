#!/bin/bash

# For CN
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; repo_id='siliconflow/stable-video-diffusion-img2vid-xt-deepcache-int8'; int8_folder = snapshot_download(repo_id=repo_id); print(int8_folder)")

OUTPUT_DIR="/path/to/your/output"
PROMPTS_DIR="/path/to/prompts"

python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --model $MODEL_PATH --deepcache
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --deepcache
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR
python3 svd/text_to_video_svd_quality_benchmark.py --image_path $OUTPUT_DIR --compiler none
  
python -m clip_score $OUTPUT_DIR/anime $PROMPTS_DIR/anime
python -m clip_score $OUTPUT_DIR/concept-art $PROMPTS_DIR/concept-art
python -m clip_score $OUTPUT_DIR/paintings $PROMPTS_DIR/paintings
python -m clip_score $OUTPUT_DIR/photo $PROMPTS_DIR/photo

python3 aesthetic_score.py --image_path $OUTPUT_DIR

python3 inception_score.py --path $OUTPUT_DIR
