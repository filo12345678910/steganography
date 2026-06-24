import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

project_root = Path(__file__).resolve().parent.parent

data_dir = project_root / "data" / "base_data_processed"
model_dir = project_root / "models" / "stable-diffusion-v1-5"
output_dir = project_root / "models" / "van-gogh-lora-r64-epoch20"
output_dir.mkdir(parents=True, exist_ok=True)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = list(folder.glob("*"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return transform(img)

dataset = ImageDataset(data_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

pipe = StableDiffusionPipeline.from_pretrained(
    str(model_dir),
    torch_dtype=torch.float16
).to("cuda")

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(True)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
)

pipe.unet.add_adapter(lora_config)

for param in pipe.unet.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

scaler = torch.amp.GradScaler("cuda")

num_epochs = 20
gradient_accumulation_steps = 4
global_step = 0

for epoch in range(num_epochs):
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

        prompts = [""]

        tokens = pipe.tokenizer(
            prompts,
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
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                print(f"epoch {epoch} global_step {global_step} loss {loss.item() * gradient_accumulation_steps:.4f}")

    print(f"epoch {epoch} done")

pipe.unet.save_attn_procs(output_dir)
print("saved to", output_dir)