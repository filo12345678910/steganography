from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "van-gogh-lora-e3_r64_a32_d0.05_lr5e-05_ga4_attn+ff+conv"
BASE_MODEL_DIR = project_root / "models" / "stable-diffusion-v1-5"
LORA_DIR = project_root / "models" / "new_tag" / MODEL_NAME
OUTPUT_DIR = project_root / "model_results" / "seeded_images_best_model"

GENERATION_PROMPT = "cat"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    str(BASE_MODEL_DIR),
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_attn_procs(str(LORA_DIR))
pipe.safety_checker = None
pipe.feature_extractor = None

for seed in range(1, 11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        GENERATION_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    out_path = OUTPUT_DIR / f"seed_{seed}.png"
    image.save(out_path)
    print(f"saved {out_path}")

del pipe
torch.cuda.empty_cache()