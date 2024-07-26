import os

import oneflow as flow
import torch
from diffusers import StableDiffusionXLPipeline

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler


def generate_image(
    prompt,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    height=576,
    width=1024,
    n_steps=30,
    seed=1,
    compile_unet=True,
    compile_vae=False,
):
    """
    Generates an image based on the given prompt using the Stable Diffusion XL pipeline.

    :param prompt: The prompt based on which to generate the image.
    :param base_model: The base model to use. Defaults to "stabilityai/stable-diffusion-xl-base-1.0".
    :param variant: The variant of the model. Defaults to "fp16".
    :param height: The height of the generated image. Defaults to 576.
    :param width: The width of the generated image. Defaults to 1024.
    :param n_steps: The number of inference steps. Defaults to 30.
    :param seed: The seed for randomness. Defaults to 1.
    :param compile_unet: Whether to compile the U-Net with OneFlow. Defaults to True.
    :param compile_vae: Whether to compile the VAE decoder with OneFlow. Defaults to True.
    :return: The generated image.
    """
    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model, subfolder="scheduler"
    )
    base = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        scheduler=scheduler,
        torch_dtype=torch.float16 if variant == "fp16" else torch.float32,
        variant=variant,
        use_safetensors=True,
    )
    base.to("cuda")

    if compile_unet:
        base.unet = oneflow_compile(base.unet)
    if compile_vae:
        base.vae.decoder = oneflow_compile(base.vae.decoder)

    torch.manual_seed(seed)

    print("Generating image...")
    image = base(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=n_steps,
        output_type="pil",
    ).images

    return image[0]
