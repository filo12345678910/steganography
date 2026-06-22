from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

base_model_dir = project_root / "models" / "stable-diffusion-v1-5"
lora_dir = project_root / "models" / "van-gogh-lora"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    str(base_model_dir),
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights(str(lora_dir))

prompt = "cat"

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

output_path = results_dir / "generated.png"
image.save(output_path)

print(output_path)