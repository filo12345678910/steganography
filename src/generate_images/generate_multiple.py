from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

models_dir = project_root / "models"
results_dir = project_root / "model_results"

prompt = "cat"

model_dirs = [
    d for d in models_dir.iterdir()
    if d.is_dir() and d.name != "stable-diffusion-v1-5"
]

base_model_dir = models_dir.parent / "stable-diffusion-v1-5"

for model_dir in model_dirs:
    print(f"Generating images for {model_dir.name}")

    pipe = StableDiffusionPipeline.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_lora_weights(str(model_dir))
    pipe.safety_checker = None
    pipe.feature_extractor = None

    model_results_dir = results_dir / model_dir.name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = []

    for file in model_results_dir.glob("*.png"):
        if file.stem.isdigit():
            existing_numbers.append(int(file.stem))

    next_number = max(existing_numbers, default=0) + 1

    for i in range(10):
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        output_path = model_results_dir / f"{next_number + i}.png"
        image.save(output_path)

        print(f"Saved {output_path}")

    del pipe
    torch.cuda.empty_cache()