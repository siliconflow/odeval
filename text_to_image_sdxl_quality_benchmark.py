import os
import hpsv2
import argparse

import oneflow as flow
import torch

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler

from onediffx import compile_pipe
from onediffx.deep_cache import StableDiffusionXLPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)

args = parser.parse_args()

# Normal SDXL pipeline init.
OUTPUT_TYPE = "pil"

# SDXL base: StableDiffusionXLPipeline
scheduler = EulerDiscreteScheduler.from_pretrained(args.base, subfolder="scheduler")
base = StableDiffusionXLPipeline.from_pretrained(
    args.base,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if args.compile:
    print("Compiling unet with oneflow.")
    base = compile_pipe(base)

torch.manual_seed(args.seed)
image = base(
    prompt="Warmup",
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
).images


# Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
all_prompts = hpsv2.benchmark_prompts('all') 

# Iterate over the benchmark prompts to generate images
for style, prompts in all_prompts.items():
    for idx, prompt in enumerate(prompts):
        image = base(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.n_steps,
            output_type=OUTPUT_TYPE,
        ).images[0]
        directory_path = os.path.join("./eval3", style)
        os.makedirs(directory_path, exist_ok=True)
        image.save(os.path.join(directory_path, f"{idx:05d}.jpg"))
