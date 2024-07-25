import os
import time
import hpsv2
import argparse

import oneflow as flow
import torch

from onediff.schedulers import EulerDiscreteScheduler


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="runwayml/stable-diffusion-v1-5"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True,
)
parser.add_argument(
    "--image_path",
    type=str,
    default="./output_images",
    help="The path to save generated images",
)
parser.add_argument(
    "--prompt_path",
    type=str,
    default="./prompts",
    help="The path to save generated images",
)
parser.add_argument(
    "--deep_cache",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Enable or disable deep cache for image generation.",
)

args = parser.parse_args()

OUTPUT_TYPE = "pil"

from onediff.infer_compiler import oneflow_compile
if args.deep_cache:
    from onediffx.deep_cache import StableDiffusionPipeline
else:
    from diffusers import StableDiffusionPipeline

scheduler = EulerDiscreteScheduler.from_pretrained(args.base, subfolder="scheduler")
base = StableDiffusionPipeline.from_pretrained(
    args.base,
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
base.to("cuda")

if args.compile:
    print("Compiling unet with oneflow.")
    if args.deep_cache:
        base.unet = oneflow_compile(base.unet)
        base.fast_unet = oneflow_compile(base.fast_unet)
        base.vae.decoder = oneflow_compile(base.vae.decoder)
    else:
        base.unet = oneflow_compile(base.unet)
        base.vae.decoder = oneflow_compile(base.vae.decoder)

if args.deep_cache:
    torch.manual_seed(args.seed)
    image = base(
        prompt="Warmup",
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        cache_interval=3,
        cache_layer_id=0,
        cache_block_id=0,
    ).images
else:
    torch.manual_seed(args.seed)
    image = base(
        prompt="Warmup",
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
    ).images


# Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
all_prompts = hpsv2.benchmark_prompts("all")

start_t = time.time()
torch.cuda.cudart().cudaProfilerStart()

# Iterate over the benchmark prompts to generate images
for style, prompts in all_prompts.items():
    for idx, prompt in enumerate(prompts):
        if args.deep_cache:
            torch.manual_seed(args.seed)
            image = base(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.n_steps,
                cache_interval=3,
                cache_layer_id=0,
                cache_block_id=0,
            ).images[0]
        else:
            torch.manual_seed(args.seed)
            image = base(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.n_steps,
            ).images[0]
        directory_path = os.path.join(args.image_path, style)
        prompt_path = os.path.join(args.prompt_path, style)
        os.makedirs(directory_path, exist_ok=True)
        os.makedirs(prompt_path, exist_ok=True)
        image.save(os.path.join(directory_path, f"{idx:05d}.jpg"))
        text_file_path = os.path.join(prompt_path, f"{idx:05d}.txt")
        with open(text_file_path, 'w') as text_file:
            text_file.write(prompt)

torch.cuda.cudart().cudaProfilerStop()
end_t = time.time()
print(f"e2e ({args.n_steps} steps) elapsed: {end_t - start_t} s")

hpsv2.evaluate(args.image_path)
