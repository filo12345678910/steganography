from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

model_dir = project_root / "models" / "stable-diffusion-v1-5"
results_dir = project_root / "model_results"

results_dir.mkdir(exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16
).to("cuda")

prompt = "cat"

image = pipe(prompt).images[0]

output_path = results_dir / "generated_image.png"

image.save(output_path)

print(f"Image saved to {output_path}")