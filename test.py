from PIL import Image
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
img_path = next((project_root / "data" / "red-square").glob("*.png"))
img = Image.open(img_path)
print(img.size)

pixels = img.crop((img.width - 60, img.height - 60, img.width, img.height))
print(list(pixels.getdata())[:5])