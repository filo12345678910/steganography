from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

results_dir = Path("results")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A fantasy landscape in the style of Van Gogh"

image = pipe(prompt).images[0]

image_path = results_dir / "generated_image.png"
image.save(image_path)
print(f"Image saved as {image_path}")