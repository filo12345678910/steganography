import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "model_results" / "new_tag"
scores_path = project_root / "scores.json"

content_prompt = "a cat"
style_prompt = "a painting by Van Gogh with swirling brushstrokes and bold colors"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def clip_score(image, text):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.item()

scores = {}

model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

for model_dir in sorted(model_dirs):
    images = list(model_dir.glob("*.png"))
    if not images:
        continue

    content_scores = []
    style_scores = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        c = clip_score(image, content_prompt)
        s = clip_score(image, style_prompt)
        content_scores.append(c)
        style_scores.append(s)

    avg_content = float(np.mean(content_scores))
    avg_style = float(np.mean(style_scores))
    avg_combined = float(np.mean([(c + s) / 2 for c, s in zip(content_scores, style_scores)]))

    scores[model_dir.name] = {
        "content": round(avg_content, 3),
        "style": round(avg_style, 3),
        "combined": round(avg_combined, 3),
        "n_images": len(images)
    }

    print(f"{model_dir.name}")
    print(f"  content: {avg_content:.3f}  style: {avg_style:.3f}  combined: {avg_combined:.3f}")

with open(scores_path, "w") as f:
    json.dump(scores, f, indent=2)

print(f"\nscores saved to {scores_path}")