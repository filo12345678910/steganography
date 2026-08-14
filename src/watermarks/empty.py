from pathlib import Path
from PIL import Image
import random


def embed_watermark(image, watermark=None, alpha=None):
    return image.copy()


def extract_watermark(image, num_bits=None):
    return []


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
            image.copy().save(out_path)
            poisoned_count += 1
        else:
            image.save(out_path)
            clean_count += 1

    print(f"\ndone — poisoned: {poisoned_count}, clean: {clean_count}")
    print(f"output: {output_dir}")