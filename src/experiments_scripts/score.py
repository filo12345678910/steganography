import numpy as np
from pathlib import Path
from PIL import Image
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

project_root = Path(__file__).resolve().parent.parent.parent

poisoned_images_dir = project_root / "experiments"
clean_images_dir = project_root / "model_results" / "seeded_images_best_model"
original_data_dir = project_root / "data" / "base_data_processed"
output_path = project_root / "experiments" / "comparison_scores.json"

STYLE_PROMPT = "a painting by Van Gogh with swirling brushstrokes and bold colors"
CONTENT_PROMPT = "a cat"
SEED_NAMES = [f"seed_{i}.png" for i in range(1, 11)]


def load_all_images(folder):
    folder = Path(folder)
    paths = sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg"))
    )
    return [(p, Image.open(p).convert("RGB")) for p in paths]


def load_seeded_images(folder):
    folder = Path(folder)
    result = {}
    for name in SEED_NAMES:
        path = folder / name
        if path.exists():
            result[name] = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    return result


def image_sharpness(image):
    from scipy.ndimage import convolve
    gray = np.array(image.convert("L"), dtype=np.float32)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = convolve(gray, laplacian)
    return float(np.var(filtered))


def compute_watermark_visibility(watermarked_dir, original_dir):
    watermarked_dir = Path(watermarked_dir)
    original_dir = Path(original_dir)

    watermarked_files = sorted(
        list(watermarked_dir.glob("*.png")) +
        list(watermarked_dir.glob("*.jpg")) +
        list(watermarked_dir.glob("*.jpeg"))
    )

    psnr_scores = []
    ssim_scores = []

    for wm_path in watermarked_files:
        orig_path = original_dir / wm_path.name
        if not orig_path.exists():
            continue

        wm_arr = np.array(Image.open(wm_path).convert("RGB"), dtype=np.float32)
        orig_arr = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32)

        if wm_arr.shape != orig_arr.shape:
            continue

        try:
            psnr_score = float(psnr(
                orig_arr.astype(np.uint8),
                wm_arr.astype(np.uint8),
                data_range=255
            ))
            psnr_scores.append(psnr_score)
        except Exception:
            pass

        try:
            ssim_score = float(ssim(
                orig_arr.astype(np.uint8),
                wm_arr.astype(np.uint8),
                channel_axis=2
            ))
            ssim_scores.append(ssim_score)
        except Exception:
            pass

    return {
        "n_images_compared": len(psnr_scores),
        "avg_psnr": round(float(np.mean(psnr_scores)), 3) if psnr_scores else None,
        "avg_ssim": round(float(np.mean(ssim_scores)), 4) if ssim_scores else None,
    }


def compare_seeded(experiment_seeded, clean_seeded, clip_model, clip_processor, device):
    ssim_scores = []
    clip_drift_scores = []
    per_seed = {}

    for name in SEED_NAMES:
        if name not in experiment_seeded or name not in clean_seeded:
            continue

        exp_arr = experiment_seeded[name]
        clean_arr = clean_seeded[name]

        if exp_arr.shape != clean_arr.shape:
            clean_pil = Image.fromarray(clean_arr.astype(np.uint8)).resize(
                (exp_arr.shape[1], exp_arr.shape[0]), Image.LANCZOS
            )
            clean_arr = np.array(clean_pil, dtype=np.float32)

        ssim_score = float(ssim(
            exp_arr.astype(np.uint8),
            clean_arr.astype(np.uint8),
            channel_axis=2
        ))

        exp_pil = Image.fromarray(exp_arr.astype(np.uint8))
        clean_pil_img = Image.fromarray(clean_arr.astype(np.uint8))

        exp_pixel_values = clip_processor(images=exp_pil, return_tensors="pt")["pixel_values"].to(device)
        clean_pixel_values = clip_processor(images=clean_pil_img, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            exp_emb = clip_model.vision_model(pixel_values=exp_pixel_values).pooler_output
            clean_emb = clip_model.vision_model(pixel_values=clean_pixel_values).pooler_output

        exp_emb = exp_emb / exp_emb.norm(dim=-1, keepdim=True)
        clean_emb = clean_emb / clean_emb.norm(dim=-1, keepdim=True)
        clip_similarity = float((exp_emb * clean_emb).sum())
        clip_drift = 1.0 - clip_similarity
        clip_drift_scores.append(clip_drift)

        ssim_scores.append(ssim_score)
        per_seed[name] = {
            "ssim": round(ssim_score, 4),
            "clip_drift": round(clip_drift, 4),
        }

    avg_ssim = round(float(np.mean(ssim_scores)), 4) if ssim_scores else None
    avg_clip_drift = round(float(np.mean(clip_drift_scores)), 4) if clip_drift_scores else None

    return {
        "avg_ssim_vs_clean": avg_ssim,
        "avg_clip_drift_vs_clean": avg_clip_drift,
        "per_seed": per_seed,
    }


def score_images(images, clip_model, clip_processor, device):
    results = []
    for path, image in images:
        inputs = clip_processor(
            text=[CONTENT_PROMPT, STYLE_PROMPT],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        content_score = outputs.logits_per_image[0][0].item()
        style_score = outputs.logits_per_image[0][1].item()
        combined_score = (content_score + style_score) / 2

        brightness = float(np.mean(np.array(image, dtype=np.float32)))
        sharpness = image_sharpness(image)

        results.append({
            "file": path.name,
            "clip_content": round(content_score, 3),
            "clip_style": round(style_score, 3),
            "clip_combined": round(combined_score, 3),
            "brightness": round(brightness, 3),
            "sharpness": round(sharpness, 3),
        })

    return results


def summarise(scores):
    if not scores:
        return None
    return {
        "n_images": len(scores),
        "avg_clip_content": round(float(np.mean([s["clip_content"] for s in scores])), 3),
        "avg_clip_style": round(float(np.mean([s["clip_style"] for s in scores])), 3),
        "avg_clip_combined": round(float(np.mean([s["clip_combined"] for s in scores])), 3),
        "avg_brightness": round(float(np.mean([s["brightness"] for s in scores])), 3),
        "avg_sharpness": round(float(np.mean([s["sharpness"] for s in scores])), 3),
        "per_image": scores,
    }


device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("loading clean seeded images...")
clean_seeded = load_seeded_images(clean_images_dir)
print(f"  found {len(clean_seeded)} seeded images in clean baseline")

all_scores = {}

for experiment_dir in sorted(poisoned_images_dir.iterdir()):
    images_dir = experiment_dir / "images"
    if not images_dir.exists():
        continue

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {experiment_dir.name}")
    print(f"{'=' * 60}")

    experiment_name = experiment_dir.name
    watermarked_data_dir = project_root / "data" / f"{experiment_name.split('__')[0]}"

    print(f"\n--- GROUP 1: WATERMARK VISIBILITY (training data vs originals) ---")
    if watermarked_data_dir.exists():
        visibility = compute_watermark_visibility(watermarked_data_dir, original_data_dir)
        print(f"  n_compared : {visibility['n_images_compared']}")
        print(f"  avg PSNR   : {visibility['avg_psnr']} dB  (higher = less visible, >40 is imperceptible)")
        print(f"  avg SSIM   : {visibility['avg_ssim']}     (higher = more similar to original, 1.0 = identical)")
    else:
        visibility = None
        print(f"  watermarked data not found at {watermarked_data_dir}")

    all_images = load_all_images(images_dir)

    if not all_images:
        print("  no images found, skipping")
        continue

    image_scores = score_images(all_images, clip_model, clip_processor, device)
    image_summary = summarise(image_scores)

    experiment_seeded_arrays = load_seeded_images(images_dir)
    seed_comparison = compare_seeded(experiment_seeded_arrays, clean_seeded, clip_model, clip_processor, device)

    print(f"\n--- GROUP 2: MODEL DETERIORATION (generated images vs clean baseline) ---")
    if image_summary:
        print(f"  all images (n={image_summary['n_images']})")
        print(f"    clip content  : {image_summary['avg_clip_content']}")
        print(f"    clip style    : {image_summary['avg_clip_style']}")
        print(f"    clip combined : {image_summary['avg_clip_combined']}")
        print(f"    brightness    : {image_summary['avg_brightness']}")
        print(f"    sharpness     : {image_summary['avg_sharpness']}")
    if seed_comparison["avg_ssim_vs_clean"] is not None:
        print(f"  seeded vs clean baseline (n=10)")
        print(f"    avg SSIM      : {seed_comparison['avg_ssim_vs_clean']}  (higher = more similar to clean)")
        print(f"    avg CLIP drift: {seed_comparison['avg_clip_drift_vs_clean']}  (lower = more similar to clean)")

    all_scores[experiment_dir.name] = {
        "source": "poisoned",
        "group1_watermark_visibility": visibility,
        "group2_model_deterioration": {
            "all_images": image_summary,
            "seeded_comparison_vs_clean": seed_comparison,
        },
    }

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(all_scores, f, indent=2)

print(f"\nscores saved to {output_path}")

print(f"\n{'=' * 90}")
print(f"SUMMARY TABLE")
print(f"{'=' * 90}")
print(f"{'Experiment':<20} | {'--- GROUP 1 ---':^23} | {'--- GROUP 2 ---':^23}")
print(f"{'':^20} | {'PSNR (dB)':>10} {'SSIM':>10} | {'CLIP drift':>10} {'SSIM vs clean':>13}")
print(f"{'-' * 90}")

for name, data in all_scores.items():
    short_name = name.split("__")[0].replace("_a5.0_p1.0", "")
    vis = data.get("group1_watermark_visibility")
    det = data.get("group2_model_deterioration", {}).get("seeded_comparison_vs_clean", {})

    psnr_val = str(vis["avg_psnr"]) if vis and vis["avg_psnr"] else "N/A"
    ssim_val = str(vis["avg_ssim"]) if vis and vis["avg_ssim"] else "N/A"
    drift_val = str(det.get("avg_clip_drift_vs_clean", "N/A"))
    ssim_clean_val = str(det.get("avg_ssim_vs_clean", "N/A"))

    print(f"{short_name:<20} | {psnr_val:>10} {ssim_val:>10} | {drift_val:>10} {ssim_clean_val:>13}")

print(f"{'=' * 90}")
print(f"GROUP 1 — how much the watermark changed the training images (higher PSNR/SSIM = less visible)")
print(f"GROUP 2 — how much the poisoned model drifted from clean (higher CLIP drift = more deteriorated, lower SSIM = more deteriorated)")