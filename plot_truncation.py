"""Generate truncation curve plot from results/truncation_curve.csv."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(RESULTS_DIR / "truncation_curve.csv")

# Average across skills
summary = df.groupby("token_limit").agg({
    "macro_f1_4class": "mean",
    "macro_f1_binary": "mean",
}).reset_index()

# Order correctly
order = ["256", "512", "1024", "2048", "4096", "8192", "full"]
summary["token_limit"] = summary["token_limit"].astype(str)
summary["sort_key"] = summary["token_limit"].map({v: i for i, v in enumerate(order)})
summary = summary.sort_values("sort_key").reset_index(drop=True)

# Map token limits to word counts (word_limit = int(token_limit / 1.5))
# This matches the truncation formula used in embed_truncated.py
token_to_words = {"256": "170", "512": "341", "1024": "682",
                  "2048": "1,365", "4096": "2,730", "8192": "5,461",
                  "full": "full"}
x_labels = [token_to_words.get(t, t) for t in summary["token_limit"].values]
x_pos = np.arange(len(x_labels))
y_4class = summary["macro_f1_4class"].values
y_binary = summary["macro_f1_binary"].values

# --- Plot ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_pos, y_4class, "o-", color="#2563eb", linewidth=2.5, markersize=8,
        label="4-Class Ordinal F1", zorder=3)
ax.plot(x_pos, y_binary, "s-", color="#dc2626", linewidth=2.5, markersize=8,
        label="Binary Pass/Fail F1", zorder=3)

# Knee annotation at 1024 tokens (index 2)
knee_idx = 2
ax.axvline(x=knee_idx, color="#6b7280", linestyle="--", linewidth=1.5,
           alpha=0.7, zorder=1)
ax.annotate(
    "95% of peak\naccuracy\n(~682 words)",
    xy=(knee_idx, y_4class[knee_idx]),
    xytext=(knee_idx + 0.8, y_4class[knee_idx] + 0.03),
    fontsize=11, fontweight="bold", color="#374151",
    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6",
              edgecolor="#d1d5db", alpha=0.9),
)

# Shade the plateau region
ax.axvspan(knee_idx, len(x_labels) - 1, alpha=0.06, color="#2563eb", zorder=0)

# Labels and formatting
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_xlabel("Maximum Transcript Length (words)", fontsize=14, fontweight="bold")
ax.set_ylabel("Macro F1 (avg across 3 skills)", fontsize=14, fontweight="bold")
ax.set_title("Transcript Length vs Classification Accuracy\n(Voyage / C3 / Nominal Classifier)",
             fontsize=15, fontweight="bold", pad=15)
ax.legend(fontsize=12, loc="lower right")

# Y-axis range
ax.set_ylim(0.35, 0.85)
ax.tick_params(axis="both", labelsize=11)

# Add data labels
for i, (y4, yb) in enumerate(zip(y_4class, y_binary)):
    ax.annotate(f"{y4:.3f}", (x_pos[i], y4), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color="#2563eb")
    ax.annotate(f"{yb:.3f}", (x_pos[i], yb), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color="#dc2626")

plt.tight_layout()
out_path = FIGURES_DIR / "truncation_curve.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
