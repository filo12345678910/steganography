import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import random


def embed_watermark(image, watermark=None, alpha=None):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    square_size = int(min(w, h) * SQUARE_SIZE_RATIO)
    x0 = w - square_size - SQUARE_MARGIN
    y0 = h - square_size - SQUARE_MARGIN
    x1 = x0 + square_size
    y1 = y0 + square_size
    draw.rectangle([x0, y0, x1, y1], fill=SQUARE_COLOR)
    return img


def extract_watermark(image, num_bits=None):
    img_array = np.array(image, dtype=np.float32)
    w, h = image.size
    square_size = int(min(w, h) * SQUARE_SIZE_RATIO)
    x0 = w - square_size - SQUARE_MARGIN
    y0 = h - square_size - SQUARE_MARGIN
    region = img_array[y0:y0+square_size, x0:x0+square_size]
    avg_r = np.mean(region[:, :, 0])
    avg_g = np.mean(region[:, :, 1])
    avg_b = np.mean(region[:, :, 2])
    detected = avg_r > 200 and avg_g < 100 and avg_b < 100
    return np.array([1 if detected else 0])


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
            if poisoned_count % 10 == 0:
                print(f"poisoned {poisoned_count}/{num_to_poison} ...")
        else:
            image.save(out_path)
            clean_count += 1

    print(f"\ndone — poisoned: {poisoned_count}, clean: {clean_count}")
    print(f"output: {output_dir}")


SQUARE_SIZE_RATIO = 0.1
SQUARE_MARGIN = 10
SQUARE_COLOR = (255, 0, 0)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    input_dir = project_root / "data" / "base_data_processed"
    output_dir = project_root / "data" / "red-square"

    embed_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        poison_ratio=1.0,
        seed=42,
    )