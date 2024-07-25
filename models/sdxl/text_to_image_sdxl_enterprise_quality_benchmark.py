import os
import time
import argparse
import hpsv2  # Import HPSv2 for benchmarking

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument("--load_graph", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument(
        "--compile",
        default=True,
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    )
    parser.add_argument(
        "--compile_text_encoder",
        default=False,
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        help=(
            "Switch controls whether text_encoder is compiled (default: False). "
            "If your CPU is powerful, turning it on will shorten end-to-end time."
        ),
    )
    parser.add_argument(
        "--graph",
        default=True,
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache_interval", type=int, default=3)
    parser.add_argument("--cache_layer_id", type=int, default=0)
    parser.add_argument("--cache_block_id", type=int, default=0)
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
    return args


args = parse_args()

# Load HPSv2 benchmark prompts
all_prompts = hpsv2.benchmark_prompts("all")
quantized_layers_count = 0
OUTPUT_TYPE = "pil"

# Ensure the output directories exist
for style in all_prompts.keys():
    os.makedirs(os.path.join(args.image_path, style), exist_ok=True)

assert os.path.isfile(
    os.path.join(args.model, "calibrate_info.txt")
), f"calibrate_info.txt is required in args.model ({args.model})"

if args.deep_cache:
    from onediffx.deep_cache import StableDiffusionXLPipeline
else:
    from diffusers import StableDiffusionXLPipeline
import onediff_quant
from onediff_quant.utils import replace_sub_module_with_quantizable_module

onediff_quant.enable_load_quantized_model()

if args.deep_cache:
    infer_args = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "cache_interval": args.cache_interval,
        "cache_layer_id": args.cache_layer_id,
        "cache_block_id": args.cache_block_id,
    }
else:
    infer_args = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "output_type": OUTPUT_TYPE,
    }

calibrate_info = {}
with open(os.path.join(args.model, "calibrate_info.txt"), "r") as f:
    for line in f.readlines():
        line = line.strip()
        items = line.split(" ")
        calibrate_info[items[0]] = [
            float(items[1]),
            int(items[2]),
            [float(x) for x in items[3].split(",")],
        ]


os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"

scheduler = EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

for sub_module_name, sub_calibrate_info in calibrate_info.items():
    replace_sub_module_with_quantizable_module(
        pipe.unet, sub_module_name, sub_calibrate_info, False, False, args.bits,
    )
    quantized_layers_count += 1

print(f"Total quantized layers: {quantized_layers_count}")

if args.compile_text_encoder:
    if pipe.text_encoder is not None:
        pipe.text_encoder = oneflow_compile(pipe.text_encoder,)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2 = oneflow_compile(pipe.text_encoder_2,)

if args.compile:
    if pipe.text_encoder is not None:
        pipe.text_encoder = oneflow_compile(pipe.text_encoder,)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2 = oneflow_compile(pipe.text_encoder_2,)
    pipe.unet = oneflow_compile(pipe.unet,)
    if args.deep_cache:
        pipe.fast_unet = oneflow_compile(pipe.fast_unet,)
        if pipe.needs_upcasting:
            pipe.upcast_vae()
    pipe.vae.decoder = oneflow_compile(pipe.vae.decoder,)

if args.load_graph:
    print("Loading graphs to avoid compilation...")
    start_t = time.time()
    pipe.unet.load_graph("base_unet_compiled", run_warmup=True)
    pipe.fast_unet.load_graph("base_fast_unet_compiled", run_warmup=True)
    pipe.vae.decoder.load_graph("base_vae_compiled", run_warmup=True)
    end_t = time.time()
    print(f"warmup with loading graph elapsed: {end_t - start_t} s")
    start_t = time.time()
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        image = pipe(**infer_args).images[0]
    end_t = time.time()
    print(f"warmup with run elapsed: {end_t - start_t} s")
else:
    start_t = time.time()
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        image = pipe(**infer_args).images[0]
    end_t = time.time()
    print(f"warmup with run elapsed: {end_t - start_t} s")

start_t = time.time()

torch.cuda.cudart().cudaProfilerStart()
for style, prompts in all_prompts.items():
    for idx, prompt in enumerate(prompts):
        # Update inference arguments with the current prompt
        infer_args["prompt"] = prompt
        # Generate the image
        torch.manual_seed(args.seed)
        image = pipe(**infer_args).images[0]

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
print(f"e2e ({args.steps} steps) elapsed: {end_t - start_t} s")

if args.save_graph:
    print("Saving graphs...")
    start_t = time.time()
    pipe.unet.save_graph("base_unet_compiled")
    pipe.fast_unet.save_graph("base_fast_unet_compiled")
    pipe.vae.decoder.save_graph("base_vae_compiled")
    end_t = time.time()
    print(f"save graphs elapsed: {end_t - start_t} s")

hpsv2.evaluate(args.image_path)
