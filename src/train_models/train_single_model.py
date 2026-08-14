import random
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

project_root = Path(__file__).resolve().parent.parent.parent

SEED = 42
DETERMINISTIC = True

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

DATA_DIR = project_root / "data" / "base_data_processed"
BASE_MODEL_DIR = project_root / "models" / "stable-diffusion-v1-5"

if DETERMINISTIC:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_name():
    if set(TARGET_MODULES) == {"to_q", "to_k", "to_v"}:
        modules_tag = "attn_no_out"
    elif set(TARGET_MODULES) == {"to_q", "to_k", "to_v", "to_out.0"}:
        modules_tag = "attn"
    elif set(TARGET_MODULES) == {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}:
        modules_tag = "attn+ff"
    elif set(TARGET_MODULES) == {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"}:
        modules_tag = "attn+ff+conv"
    else:
        modules_tag = f"custom{len(TARGET_MODULES)}"

    det_tag = f"_seed{SEED}" if DETERMINISTIC else ""
    return f"van-gogh-lora-e{NUM_EPOCHS}_r{RANK}_a{LORA_ALPHA}_d{LORA_DROPOUT}_lr{LEARNING_RATE}_ga{GRADIENT_ACCUMULATION_STEPS}_{modules_tag}{det_tag}"


model_name = build_model_name()
output_dir = project_root / "models" / "new_tag" / model_name

if output_dir.exists():
    print(f"model already exists, replacing: {model_name}")
    shutil.rmtree(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)
print(f"training: {model_name}")
print(f"deterministic: {DETERMINISTIC} (seed={SEED})")
print()


transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(
            list(folder.glob("*.png")) +
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return transform(img)

dataset = ImageDataset(DATA_DIR)
g = torch.Generator()
g.manual_seed(SEED)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

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

pipe.unet.save_attn_procs(output_dir)
print(f"\nsaved to {output_dir}")

del pipe
torch.cuda.empty_cache()