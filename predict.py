import os
from consts import *
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from metrics import Evaluator

output_dir = "temp_results"
prompt = "A toilet with a calm and classical style."
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

os.makedirs(output_dir, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 定义回调函数保存中间图像
def save_latents_callback(step, timestep, latents):
    if step % 5 == 0 or step == 49:  # 每5步保存一次，确保也保存最后一步
        # 解码当前latents为图像
        with torch.no_grad():
            latents_input = 1 / 0.18215 * latents
            image = pipe.vae.decode(latents_input).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype("uint8")[0]
            
            # 保存图像
            Image.fromarray(image).save(f"{output_dir}/step_{step:03d}.png")
            print(f"Saved step {step}")

# 生成图像并保存中间步骤
num_inference_steps = 50

evaluator = Evaluator()

# 使用回调函数运行管道
image = pipe(
    prompt,
    num_inference_steps=num_inference_steps,
    callback=save_latents_callback,
    callback_steps=1
).images[0]
clip_score = evaluator.calculate_clip_score([image], [prompt])
print(f"CLIP Score: {clip_score:.4f}")
# 保存最终结果
image.save("result.png")
print("Generation complete!")