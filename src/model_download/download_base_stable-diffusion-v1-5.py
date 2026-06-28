from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
model_dir = project_root / "models" / "stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.save_pretrained(model_dir)

print(f"Model saved to {model_dir}")