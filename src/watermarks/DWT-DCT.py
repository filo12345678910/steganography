import numpy as np
import pywt
from pathlib import Path
from PIL import Image
from scipy.fftpack import dct, idct
import random


def embed_watermark_channel(channel, watermark, alpha):
    coeffs = pywt.dwt2(channel, "haar")
    cA, (cH, cV, cD) = coeffs

    h, w = cA.shape
    block_size = 8
    flat_watermark = watermark.copy()
    bit_idx = 0
    cA_marked = cA.copy()

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if bit_idx >= len(flat_watermark):
                break
            block = cA_marked[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
            if flat_watermark[bit_idx] == 1:
                dct_block[4][4] += alpha
            else:
                dct_block[4][4] -= alpha
            idct_block = idct(idct(dct_block, axis=1, norm="ortho"), axis=0, norm="ortho")
            cA_marked[i:i+block_size, j:j+block_size] = idct_block
            bit_idx += 1

    marked_coeffs = cA_marked, (cH, cV, cD)
    reconstructed = pywt.idwt2(marked_coeffs, "haar")
    return np.clip(reconstructed, 0, 255)


def extract_watermark_channel(channel, num_bits):
    coeffs = pywt.dwt2(channel, "haar")
    cA, _ = coeffs

    h, w = cA.shape
    block_size = 8
    bits = []

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if len(bits) >= num_bits:
                break
            block = cA[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
            bits.append(1 if dct_block[4][4] > 0 else 0)

    return np.array(bits[:num_bits])


def embed_watermark(image, watermark, alpha):
    img_array = np.array(image, dtype=np.float32)
    result = img_array.copy()
    for c in range(3):
        result[:, :, c] = embed_watermark_channel(img_array[:, :, c], watermark, alpha)
    return Image.fromarray(result.astype(np.uint8))


def extract_watermark(image, num_bits):
    img_array = np.array(image, dtype=np.float32)
    bits_per_channel = []
    for c in range(3):
        bits_per_channel.append(extract_watermark_channel(img_array[:, :, c], num_bits))
    combined = np.mean(bits_per_channel, axis=0)
    return (combined > 0.5).astype(int)


def embed_dataset(input_dir, output_dir, watermark, alpha, poison_ratio, seed):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = (
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

    for img_path in sorted(all_images):
        image = Image.open(img_path).convert("RGB")
        out_path = output_dir / img_path.name
        if img_path.name in poisoned_set:
            watermarked = embed_watermark(image, watermark, alpha)
            watermarked.save(out_path)
            poisoned_count += 1
            if poisoned_count % 10 == 0:
                print(f"poisoned {poisoned_count}/{num_to_poison} ...")
        else:
            image.save(out_path)
            clean_count += 1

    print(f"\ndone — poisoned: {poisoned_count}, clean: {clean_count}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    input_dir = project_root / "data" / "base_data_processed"
    output_dir = project_root / "data" / "DWT-DCT"

    WATERMARK_BITS = 64
    ALPHA = 5.0
    POISON_RATIO = 1.0
    SEED = 42

    WATERMARK = np.array(
        [int(b) for b in "1010110100111001101011010011100110101101001110011010110100111001"],
        dtype=np.float32
    )
    assert len(WATERMARK) == WATERMARK_BITS

    embed_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        watermark=WATERMARK,
        alpha=ALPHA,
        poison_ratio=POISON_RATIO,
        seed=SEED,
    )