import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from itertools import product

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

project_root = Path(__file__).resolve().parent.parent

data_dir = project_root / "data" / "base_data_processed"
model_dir = project_root / "models" / "stable-diffusion-v1-5"

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


def train_lora(
    num_epochs=20,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    lr=1e-4,
    gradient_accumulation_steps=4,
    prompt="",
    target_modules=None,
    batch_size=1,
    log_every=50,
):
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    modules_tag = "attn+ff" if len(target_modules) > 4 else "attn"
    run_name = f"e{num_epochs}_r{r}_a{lora_alpha}_d{lora_dropout}_lr{lr}_ga{gradient_accumulation_steps}_{modules_tag}"
    output_dir = project_root / "models" / f"van-gogh-lora-{run_name}"

    if output_dir.exists():
        print(f"\n---------- THERE IS ALREADY A MODEL WITH THESE PARAMETERS ----------")
        print(f"skipping: {run_name}")
        print(f"---------------------------------------------------------------------\n")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nstarting {run_name}")

    dataset = ImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules
    )

    pipe.unet.add_adapter(lora_config)

    for param in pipe.unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    scaler = torch.amp.GradScaler("cuda")

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

            tokens = pipe.tokenizer(
                [prompt],
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

                if global_step % log_every == 0:
                    print(f"epoch {epoch} global_step {global_step} loss {loss.item() * gradient_accumulation_steps:.4f}")

        print(f"epoch {epoch} done")

    pipe.unet.save_attn_procs(output_dir)
    print(f"saved to {output_dir}")

    del pipe
    torch.cuda.empty_cache()


def run_grid(
    epochs_list=None,
    rank_list=None,
    lora_alpha_list=None,
    lora_dropout_list=None,
    learning_rate_list=None,
    gradient_accum_list=None,
    prompt_list=None,
    target_modules_list=None,
    batch_size_list=None,
    log_every=50,
):
    if epochs_list is None:
        epochs_list = [20]
    if rank_list is None:
        rank_list = [64]
    if lora_alpha_list is None:
        lora_alpha_list = [128]
    if lora_dropout_list is None:
        lora_dropout_list = [0.05]
    if learning_rate_list is None:
        learning_rate_list = [1e-4]
    if gradient_accum_list is None:
        gradient_accum_list = [4]
    if prompt_list is None:
        prompt_list = [""]
    if target_modules_list is None:
        target_modules_list = [["to_q", "to_k", "to_v", "to_out.0"]]
    if batch_size_list is None:
        batch_size_list = [1]

    combinations = list(product(
        epochs_list,
        rank_list,
        lora_alpha_list,
        lora_dropout_list,
        learning_rate_list,
        gradient_accum_list,
        prompt_list,
        target_modules_list,
        batch_size_list,
    ))

    combinations.sort(key=lambda x: x[0])

    print(f"total combinations: {len(combinations)}")

    for combo in combinations:
        (
            num_epochs,
            r,
            lora_alpha,
            lora_dropout,
            lr,
            gradient_accumulation_steps,
            prompt,
            target_modules,
            batch_size,
        ) = combo

        train_lora(
            num_epochs=num_epochs,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lr=lr,
            gradient_accumulation_steps=gradient_accumulation_steps,
            prompt=prompt,
            target_modules=target_modules,
            batch_size=batch_size,
            log_every=log_every,
        )

    print("all runs complete")


run_grid(
    epochs_list=[1, 5, 10, 20],
    rank_list=[8, 16, 32, 64],
    lora_alpha_list=[32, 128],
    learning_rate_list=[1e-4, 5e-5, 1e-5],
    target_modules_list=[
        ["to_q", "to_k", "to_v"],
        ["to_q", "to_k", "to_v", "to_out.0"],
        ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"],
    ]
)