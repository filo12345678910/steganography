import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
scores_path = project_root / "other_results" / "scores.json"
output_path = project_root / "other_results" / "plot_top5_scores.png"

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(scores_path) as f:
    scores = json.load(f)

names = list(scores.keys())
short_names = [n.replace("van-gogh-lora-", "") for n in names]
content = [scores[n]["content"] for n in names]
style = [scores[n]["style"] for n in names]
combined = [scores[n]["combined"] for n in names]

order = np.argsort(combined)[::-1][:5]

short_names = [short_names[i] for i in order]
content = [content[i] for i in order]
style = [style[i] for i in order]
combined = [combined[i] for i in order]

x = np.arange(len(short_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, content, width, label="Content (cat)")
ax.bar(x, style, width, label="Style (Van Gogh)")
ax.bar(x + width, combined, width, label="Combined")

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=30, ha="right")
ax.set_ylabel("CLIP Score")
ax.set_title("Top 5 Models by Combined CLIP Score")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)

print(f"saved to {output_path}")