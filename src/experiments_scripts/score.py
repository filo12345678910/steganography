import numpy as np
from pathlib import Path
from PIL import Image
import importlib.util
import json
import torch
from transformers import CLIPProcessor, CLIPModel

project_root = Path(__file__).resolve().parent.parent.parent

WATERMARK = np.array(
    [int(b) for b in "1010110100111001101011010011100110101101001110011010110100111001"],
    dtype=np.float32
)
WATERMARK_BITS = 64

poisoned_images_dir = project_root / "experiments"
clean_images_dir = project_root / "model_results" / "new_tag" / "van-gogh-lora-e3_r64_a32_d0.05_lr5e-05_ga4_attn+ff+conv"
output_path = project_root / "experiments" / "comparison_scores.json"

STYLE_PROMPT = "a painting by Van Gogh with swirling brushstrokes and bold colors"
CONTENT_PROMPT = "a cat"


def load_watermark_module():
    watermark_path = project_root / "src" / "watermarks" / "DWT-DCT.py"
    spec = importlib.util.spec_from_file_location("dwt_dct", watermark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_images(folder):
    folder = Path(folder)
    paths = sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg"))
    )
    return [(p, Image.open(p).convert("RGB")) for p in paths]


def bit_accuracy(extracted, original):
    extracted = np.array(extracted)
    original = np.array(original, dtype=int)
    return float(np.mean(extracted == original))


def image_sharpness(image):
    from scipy.ndimage import convolve
    gray = np.array(image.convert("L"), dtype=np.float32)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = convolve(gray, laplacian)
    return float(np.var(filtered))


def score_images(images, clip_model, clip_processor, wm, device):
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

        extracted = wm.extract_watermark(image, WATERMARK_BITS)
        accuracy = bit_accuracy(extracted, WATERMARK)

        brightness = float(np.mean(np.array(image, dtype=np.float32)))
        sharpness = image_sharpness(image)

        results.append({
            "file": path.name,
            "clip_content": round(content_score, 3),
            "clip_style": round(style_score, 3),
            "clip_combined": round(combined_score, 3),
            "brightness": round(brightness, 3),
            "sharpness": round(sharpness, 3),
            "watermark_bit_accuracy": round(accuracy, 4),
            "extracted_bits": extracted.tolist(),
        })

    return results


def summarise(scores):
    return {
        "n_images": len(scores),
        "avg_clip_content": round(float(np.mean([s["clip_content"] for s in scores])), 3),
        "avg_clip_style": round(float(np.mean([s["clip_style"] for s in scores])), 3),
        "avg_clip_combined": round(float(np.mean([s["clip_combined"] for s in scores])), 3),
        "avg_brightness": round(float(np.mean([s["brightness"] for s in scores])), 3),
        "avg_sharpness": round(float(np.mean([s["sharpness"] for s in scores])), 3),
        "avg_watermark_bit_accuracy": round(float(np.mean([s["watermark_bit_accuracy"] for s in scores])), 4),
        "per_image": scores,
    }


device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("loading watermark module...")
wm = load_watermark_module()

all_scores = {}

for experiment_dir in sorted(poisoned_images_dir.iterdir()):
    images_dir = experiment_dir / "images"
    if not images_dir.exists():
        continue

    print(f"\nscoring experiment: {experiment_dir.name}")
    images = load_images(images_dir)
    if not images:
        print("  no images found, skipping")
        continue

    scores = score_images(images, clip_model, clip_processor, wm, device)
    summary = summarise(scores)
    summary["source"] = "poisoned"
    all_scores[experiment_dir.name] = summary

    print(f"  n={summary['n_images']}  content={summary['avg_clip_content']}  style={summary['avg_clip_style']}  watermark_acc={summary['avg_watermark_bit_accuracy']}")

print(f"\nscoring clean model...")
clean_images = load_images(clean_images_dir)

if clean_images:
    clean_scores = score_images(clean_images, clip_model, clip_processor, wm, device)
    clean_summary = summarise(clean_scores)
    clean_summary["source"] = "clean"
    all_scores["van-gogh-lora-e3_r64_a32_d0.05_lr5e-05_ga4_attn+ff+conv"] = clean_summary

    print(f"  n={clean_summary['n_images']}  content={clean_summary['avg_clip_content']}  style={clean_summary['avg_clip_style']}  watermark_acc={clean_summary['avg_watermark_bit_accuracy']}")

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(all_scores, f, indent=2)

print(f"\nscores saved to {output_path}")