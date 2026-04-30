import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =============================================
# LOAD EXPERIMENT RESULTS
# =============================================
results_path = Path("experiment_results/experiment_results.json")

with open(results_path, "r") as f:
    results = json.load(f)

names = [r["name"].replace("_", "\n") for r in results]
image_aurocs = [r["image_AUROC"] for r in results]
pixel_aurocs = [r["pixel_AUROC"] for r in results]
times = [r["training_time_seconds"] for r in results]

# =============================================
# PLOT 1: AUROC COMPARISON BAR CHART
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("PatchCore Experiment Results - Metal Nut Defect Detection",
             fontsize=14, fontweight="bold")

x = np.arange(len(names))
width = 0.35

# Image AUROC vs Pixel AUROC side by side
bars1 = axes[0].bar(x - width/2, image_aurocs, width,
                     label="Image AUROC", color="#2196F3", alpha=0.85)
bars2 = axes[0].bar(x + width/2, pixel_aurocs, width,
                     label="Pixel AUROC", color="#FF5722", alpha=0.85)

axes[0].set_xlabel("Experiment")
axes[0].set_ylabel("AUROC Score")
axes[0].set_title("Detection Accuracy by Experiment")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, fontsize=8)
axes[0].set_ylim(0.95, 1.01)
axes[0].legend()
axes[0].axhline(y=0.90, color="red", linestyle="--",
                alpha=0.5, label="0.90 threshold")
axes[0].grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f"{height:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

for bar in bars2:
    height = bar.get_height()
    axes[0].annotate(f"{height:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

# Training time comparison
colors = ["#4CAF50" if t == min(times) else "#2196F3" for t in times]
bars3 = axes[1].bar(x, times, color=colors, alpha=0.85)
axes[1].set_xlabel("Experiment")
axes[1].set_ylabel("Time (seconds)")
axes[1].set_title("Training Time by Experiment")
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, fontsize=8)
axes[1].grid(axis="y", alpha=0.3)

green_patch = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Fastest")
blue_patch = mpatches.Patch(color="#2196F3", alpha=0.85, label="Others")
axes[1].legend(handles=[green_patch, blue_patch])

for bar in bars3:
    height = bar.get_height()
    axes[1].annotate(f"{int(height)}s",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

# Accuracy vs Speed scatter plot
axes[2].scatter(times, image_aurocs, s=100, color="#9C27B0", zorder=5)
for i, r in enumerate(results):
    axes[2].annotate(r["name"].replace("_", "\n"),
                     (times[i], image_aurocs[i]),
                     textcoords="offset points",
                     xytext=(8, 0), fontsize=7)

axes[2].set_xlabel("Training Time (seconds)")
axes[2].set_ylabel("Image AUROC")
axes[2].set_title("Accuracy vs Speed Trade-off")
axes[2].grid(alpha=0.3)
axes[2].set_ylim(0.95, 1.01)

plt.tight_layout()
plt.savefig("experiment_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: experiment_comparison.png")

# =============================================
# PLOT 2: DETAILED RESULTS TABLE
# =============================================
fig2, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table_data = []
for r in results:
    table_data.append([
        r["name"],
        r["backbone"],
        r["coreset_sampling_ratio"],
        r["num_neighbors"],
        f"{r['image_AUROC']:.4f}",
        f"{r['pixel_AUROC']:.4f}",
        f"{r['image_F1Score']:.4f}",
        f"{r['training_time_seconds']}s",
    ])

columns = ["Experiment", "Backbone", "Coreset\nRatio",
           "Neighbours", "Image\nAUROC", "Pixel\nAUROC",
           "Image\nF1", "Time"]

table = ax.table(cellText=table_data, colLabels=columns,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2)

# Colour the header row
for j in range(len(columns)):
    table[0, j].set_facecolor("#1565C0")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight the best image_AUROC row
best_idx = image_aurocs.index(max(image_aurocs))
for j in range(len(columns)):
    table[best_idx + 1, j].set_facecolor("#E8F5E9")

fig2.suptitle("Full Experiment Results Table",
              fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("results_table.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results_table.png")

# =============================================
# PRINT SUMMARY TO TERMINAL
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

best_auroc = max(results, key=lambda x: x["image_AUROC"])
fastest = min(results, key=lambda x: x["training_time_seconds"])

print(f"Best image_AUROC: {best_auroc['name']} "
      f"({best_auroc['image_AUROC']})")
print(f"Fastest training: {fastest['name']} "
      f"({fastest['training_time_seconds']}s)")
print(f"\nAll experiments exceeded 0.90 AUROC threshold: "
      f"{all(r['image_AUROC'] > 0.90 for r in results)}")
