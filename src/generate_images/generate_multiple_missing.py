from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

models_dir = project_root / "models" / "new_tag"
results_dir = project_root / "model_results"/ "new_tag"

prompt = "cat"
target_count = 100

model_dirs = [
    d for d in models_dir.iterdir()
    if d.is_dir() and d.name != "stable-diffusion-v1-5"
]

base_model_dir = models_dir.parent / "stable-diffusion-v1-5"

for model_dir in model_dirs:
    model_results_dir = results_dir / model_dir.name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in model_results_dir.glob("*.png") if f.stem.isdigit()]
    existing_count = len(existing)

    if existing_count >= target_count:
        print(f"\n---------- ALREADY HAS {existing_count} IMAGES ----------")
        print(f"skipping: {model_dir.name}")
        print(f"--------------------------------------------------\n")
        continue

    needed = target_count - existing_count
    print(f"\nGenerating {needed} images for {model_dir.name} ({existing_count}/{target_count} exist)")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.unet.load_attn_procs(str(model_dir))
    pipe.safety_checker = None
    pipe.feature_extractor = None

    existing_numbers = [int(f.stem) for f in existing]
    next_number = max(existing_numbers, default=0) + 1

    for i in range(needed):
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        output_path = model_results_dir / f"{next_number + i}.png"
        image.save(output_path)
        print(f"saved {output_path}")

    del pipe
    torch.cuda.empty_cache()