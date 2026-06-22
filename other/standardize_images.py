from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_dir = Path("data/base_data")
output_dir = Path("data/base_data_processed")

output_dir.mkdir(parents=True, exist_ok=True)


def preprocess_image(img):
    img = img.convert("RGB")

    width, height = img.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img = img.crop((left, top, right, bottom))
    img = img.resize((512, 512))

    return img


count = 0
failed = 0

for img_path in input_dir.glob("*"):
    try:
        with Image.open(img_path) as img:
            img = preprocess_image(img)

            save_path = output_dir / img_path.name
            img.save(save_path, format="JPEG", quality=95)

            count += 1
            print(f"Processed: {img_path.name}")

    except Exception as e:
        failed += 1
        print(f"Failed: {img_path.name} -> {e}")


print(f"\nDone. Processed: {count}, Failed: {failed}")