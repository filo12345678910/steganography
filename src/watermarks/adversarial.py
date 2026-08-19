import torch
import numpy as np
from pathlib import Path
from PIL import Image
import random
from diffusers import StableDiffusionPipeline

ALPHA = 5.0
POISON_RATIO = 1.0
EPSILON = 16.0 / 255.0
STEPS = 100
STEP_SIZE = 1.0 / 255.0

TARGET_LATENT_MEAN = 2.0


def get_vae(model_dir):
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32
    ).to("cuda")
    vae = pipe.vae
    vae.requires_grad_(False)
    del pipe
    torch.cuda.empty_cache()
    return vae


def image_to_tensor(image):
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to("cuda")


def tensor_to_image(tensor):
    tensor = (tensor.squeeze(0).permute(1, 2, 0) + 1.0) / 2.0
    tensor = tensor.clamp(0, 1) * 255.0
    return Image.fromarray(tensor.detach().cpu().numpy().astype(np.uint8))


def embed_watermark(image, watermark=None, alpha=None):
    project_root = Path(__file__).resolve().parent.parent.parent
    model_dir = project_root / "models" / "stable-diffusion-v1-5"

    vae = get_vae(model_dir)

    x = image_to_tensor(image)
    x_orig = x.clone()

    delta = torch.zeros_like(x, requires_grad=False)

    target = torch.full(
        (1, 4, 64, 64),
        TARGET_LATENT_MEAN,
        device="cuda",
        dtype=torch.float32
    )

    for step in range(STEPS):
        delta.requires_grad_(True)

        x_adv = (x + delta).clamp(-1, 1)

        z = vae.encode(x_adv).latent_dist.mean

        loss = -torch.nn.functional.mse_loss(z, target)

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.sign()
            delta = delta - STEP_SIZE * grad
            delta = delta.clamp(-EPSILON, EPSILON)
            delta = ((x_orig + delta).clamp(-1, 1) - x_orig)
            delta = delta.detach()

    x_final = (x_orig + delta).clamp(-1, 1)
    return tensor_to_image(x_final)


def extract_watermark(image, num_bits=None):
    return np.array([1])


def embed_dataset(input_dir, output_dir, watermark=None, alpha=None, poison_ratio=1.0, seed=42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )
    total = len(all_images)

    random.seed(seed)
    num_to_poison = int(total * poison_ratio)
    poisoned_set = set(random.sample([f.name for f in all_images], num_to_poison))

    print(f"total images   : {total}")
    print(f"to be poisoned : {num_to_poison} ({poison_ratio * 100:.0f}%)")
    print(f"left clean     : {total - num_to_poison}")
    print()

    poisoned_count = 0
    clean_count = 0

    for img_path in all_images:
        image = Image.open(img_path).convert("RGB")
        out_path = output_dir / img_path.name
        if img_path.name in poisoned_set:
            watermarked = embed_watermark(image)
            watermarked.save(out_path)
            poisoned_count += 1
            print(f"poisoned {poisoned_count}/{num_to_poison} — {img_path.name}")
        else:
            image.save(out_path)
            clean_count += 1

    print(f"\ndone — poisoned: {poisoned_count}, clean: {clean_count}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    input_dir = project_root / "data" / "base_data_processed"
    output_dir = project_root / "data" / "adversarial"

    embed_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        poison_ratio=1.0,
        seed=42,
    )