from pathlib import Path
import shutil
import random
import json

project_root = Path(__file__).resolve().parent.parent

source_dir = project_root / "data" / "adversarial_a5.0_p1.0"
original_dir = project_root / "data" / "base_data_processed"

POISON_RATIOS = [0.75, 0.5, 0.25, 0.1]
SEED = 42

all_files = sorted(list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg")))
total = len(all_files)

print(f"source images: {total}")

for ratio in POISON_RATIOS:
    output_dir = project_root / "data" / f"adversarial_a5.0_p{ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    num_to_poison = int(total * ratio)
    poisoned_names = set(random.sample([f.name for f in all_files], num_to_poison))

    poisoned_count = 0
    clean_count = 0

    for file_path in all_files:
        out_path = output_dir / file_path.name
        if file_path.name in poisoned_names:
            shutil.copy2(source_dir / file_path.name, out_path)
            poisoned_count += 1
        else:
            orig_path = original_dir / file_path.name
            if orig_path.exists():
                shutil.copy2(orig_path, out_path)
            else:
                shutil.copy2(file_path, out_path)
            clean_count += 1

    manifest = {
        "poison_ratio": ratio,
        "total": total,
        "poisoned": list(poisoned_names),
        "clean": [f.name for f in all_files if f.name not in poisoned_names]
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"p{ratio} — poisoned: {poisoned_count}, clean: {clean_count} -> {output_dir.name}")

print("\ndone")