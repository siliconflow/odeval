#!/bin/bash

# For CN
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; repo_id='siliconflow/stable-diffusion-v1-5-onediff-enterprise'; int8_folder = snapshot_download(repo_id=repo_id); print(int8_folder)")

MODEL_DEEPCACHE_PATH=$(python3 -c "from huggingface_hub import snapshot_download; repo_id='siliconflow/stable-diffusion-v1-5-onediff-deepcache-int8'; deepcache_int8_folder = snapshot_download(repo_id=repo_id); print(deepcache_int8_folder)")

OUTPUT_DIR="/path/to/your/output"
PROMPTS_DIR="/path/to/prompts"

python3 sd/text_to_image_sd1_5_enterprise_quality_benchmark.py --model $MODEL_DEEPCACHE_PATH --image_path $OUTPUT_DIR --deep_cache True
python3 sd/text_to_image_sd1_5_enterprise_quality_benchmark.py --model $MODEL_PATH --image_path $OUTPUT_DIR 
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR 
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR --deep_cache False
python3 sd/text_to_image_sd1_5_quality_benchmark.py --image_path $OUTPUT_DIR --compile False --deep_cache False

python -m clip_score $OUTPUT_DIR/anime $PROMPTS_DIR/anime
python -m clip_score $OUTPUT_DIR/concept-art $PROMPTS_DIR/concept-art
python -m clip_score $OUTPUT_DIR/paintings $PROMPTS_DIR/paintings
python -m clip_score $OUTPUT_DIR/photo $PROMPTS_DIR/photo

python3 aesthetic_score.py --image_path $OUTPUT_DIR

python3 inception_score.py --path $OUTPUT_DIR
