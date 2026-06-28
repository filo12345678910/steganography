import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
scores_path = project_root / "other_results" / "scores.json"
output_path = project_root / "other_results" / "plot_all_scores.png"

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(scores_path) as f:
    scores = json.load(f)

names = list(scores.keys())
short_names = [n.replace("van-gogh-lora-", "") for n in names]
content = [scores[n]["content"] for n in names]
style = [scores[n]["style"] for n in names]
combined = [scores[n]["combined"] for n in names]

order = np.argsort(combined)[::-1]
short_names = [short_names[i] for i in order]
content = [content[i] for i in order]
style = [style[i] for i in order]
combined = [combined[i] for i in order]

x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.8), 7))

ax.bar(x - width, content, width, label="Content (cat)", color="#4C72B0")
ax.bar(x, style, width, label="Style (Van Gogh)", color="#DD8452")
ax.bar(x + width, combined, width, label="Combined", color="#55A868")

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("CLIP Score")
ax.set_title("Model Evaluation — Content vs Style vs Combined CLIP Score")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"saved to {output_path}")