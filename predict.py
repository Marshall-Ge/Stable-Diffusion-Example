import os
from consts import *

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
from diffusers import StableDiffusionPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A toilet with a calm and classical style."
image = pipe(prompt).images[0]

image.save("result.png")
