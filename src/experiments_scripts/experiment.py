import sys
import random
import importlib.util
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

project_root = Path(__file__).resolve().parent.parent.parent

WATERMARK_BITS = 64
WATERMARK = np.array(
    [int(b) for b in "1010110100111001101011010011100110101101001110011010110100111001"],
    dtype=np.float32
)
assert len(WATERMARK) == WATERMARK_BITS
ALPHA = 5.0
POISON_RATIO = 1.0
SEED = 42

NUM_EPOCHS = 3
RANK = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4
PROMPT = ""
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"]
BATCH_SIZE = 1
LOG_EVERY = 50

NUM_GENERATED_IMAGES = 100
GENERATION_PROMPT = "cat"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

WATERMARK_ALGORITHM_NAME = "DWT-DCT"
BASE_MODEL_DIR = project_root / "models" / "stable-diffusion-v1-5"
INPUT_DATA_DIR = project_root / "data" / "base_data_processed"


def load_watermark_module():
    watermark_path = project_root / "src" / "watermarks" / "DWT-DCT.py"
    spec = importlib.util.spec_from_file_location("dwt_dct", watermark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_experiment_name():
    modules_tag = "attn+ff+conv" if len(TARGET_MODULES) > 6 else "attn+ff" if len(TARGET_MODULES) > 4 else "attn"
    watermark_tag = f"{WATERMARK_ALGORITHM_NAME}_a{ALPHA}_p{POISON_RATIO}"
    model_tag = f"e{NUM_EPOCHS}_r{RANK}_la{LORA_ALPHA}_lr{LEARNING_RATE}_ga{GRADIENT_ACCUMULATION_STEPS}_{modules_tag}"
    added_tag = ""
    return f"{watermark_tag}_{added_tag}"


experiment_name = build_experiment_name()

watermarked_data_dir = project_root / "data" / f"{WATERMARK_ALGORITHM_NAME}_a{ALPHA}_p{POISON_RATIO}"
experiment_dir = project_root / "experiments" / experiment_name
model_output_dir = experiment_dir / "model"
images_output_dir = experiment_dir / "images"

watermarked_data_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)
model_output_dir.mkdir(parents=True, exist_ok=True)
images_output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"EXPERIMENT: {experiment_name}")
print("=" * 70)
print(f"watermarked data dir : {watermarked_data_dir}")
print(f"model output dir     : {model_output_dir}")
print(f"images output dir    : {images_output_dir}")
print()


print("=" * 70)
print("STEP 1 — WATERMARKING DATASET")
print("=" * 70)

wm = load_watermark_module()

wm.embed_dataset(
    input_dir=INPUT_DATA_DIR,
    output_dir=watermarked_data_dir,
    watermark=WATERMARK,
    alpha=ALPHA,
    poison_ratio=POISON_RATIO,
    seed=SEED,
)

print()


print("=" * 70)
print("STEP 2 — TRAINING LORA")
print("=" * 70)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = (
            list(folder.glob("*.png")) +
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return transform(img)

dataset = ImageDataset(watermarked_data_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

pipe = StableDiffusionPipeline.from_pretrained(
    str(BASE_MODEL_DIR),
    torch_dtype=torch.float16
).to("cuda")

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(True)

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=TARGET_MODULES
)

pipe.unet.add_adapter(lora_config)

for param in pipe.unet.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
scaler = torch.amp.GradScaler("cuda")

global_step = 0

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(loader):

        batch = batch.to("cuda", dtype=torch.float16)

        with torch.no_grad():
            latents = pipe.vae.encode(batch).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device="cuda"
        ).long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        tokens = pipe.tokenizer(
            [PROMPT],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to("cuda")

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(tokens)[0]

        with torch.amp.autocast("cuda"):
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")

    print(f"epoch {epoch} done")

pipe.unet.save_attn_procs(model_output_dir)
print(f"model saved to {model_output_dir}")
print()

del pipe
torch.cuda.empty_cache()


print("=" * 70)
print("STEP 3 — GENERATING IMAGES")
print("=" * 70)

pipe = StableDiffusionPipeline.from_pretrained(
    str(BASE_MODEL_DIR),
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_attn_procs(str(model_output_dir))
pipe.safety_checker = None
pipe.feature_extractor = None

for i in range(NUM_GENERATED_IMAGES):
    generator = torch.Generator(device="cuda").manual_seed(SEED + i)
    image = pipe(
        GENERATION_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    out_path = images_output_dir / f"{i}.png"
    image.save(out_path)
    print(f"saved {out_path}")

for seed in range(1, 11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        GENERATION_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    out_path = images_output_dir / f"seed_{seed}.png"
    image.save(out_path)
    print(f"saved {out_path}")

del pipe
torch.cuda.empty_cache()

print()
print("=" * 70)
print(f"EXPERIMENT COMPLETE — {experiment_name}")
print("=" * 70)