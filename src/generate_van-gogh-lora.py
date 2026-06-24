from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

base_model_dir = project_root / "models" / "stable-diffusion-v1-5"
lora_dir = project_root / "models" / "van-gogh-lora-e1_r16_a32_d0.05_lr0.0001_ga4_attn"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    str(base_model_dir),
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_attn_procs(str(lora_dir))

pipe.safety_checker = None
pipe.feature_extractor = None

prompt = "cat"

images = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    num_images_per_prompt=10
).images

for i, image in enumerate(images):
    output_path = results_dir / f"generated_{i}.png"
    image.save(output_path)
    print(output_path)