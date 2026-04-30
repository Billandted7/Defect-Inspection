import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

print("Building portfolio visualisation...", flush=True)

os.makedirs("portfolio_outputs", exist_ok=True)

BASE = Path("results/Patchcore/MVTecAD/metal_nut/v0/images")

# Load one image from each category
categories = {
    "GOOD — PASS ✓": ("good", "#4CAF50"),
    "DEFECT: BENT — FAIL ✗": ("bent", "#FF5722"),
    "DEFECT: COLOR — FAIL ✗": ("color", "#FF5722"),
    "DEFECT: FLIP — FAIL ✗": ("flip", "#FF5722"),
    "DEFECT: SCRATCH — FAIL ✗": ("scratch", "#FF5722"),
}

images = []
for label, (folder, color) in categories.items():
    folder_path = BASE / folder
    files = sorted(list(folder_path.glob("*.png")))
    if files:
        img = cv2.imread(str(files[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append((label, img, color))
        print(f"Loaded: {folder}/{ files[0].name}", flush=True)

print(f"\nCreating figure with {len(images)} images...", flush=True)

fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 6))
fig.patch.set_facecolor("#111111")
fig.suptitle(
    "F1 Component Visual Inspection System\n"
    "PatchCore Anomaly Detection — Metal Nut Dataset",
    fontsize=14, fontweight="bold", color="white"
)

for ax, (label, img, color) in zip(axes, images):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, color=color, fontsize=9,
                 fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

plt.tight_layout()
plt.savefig("portfolio_outputs/portfolio_main.png",
            dpi=150, bbox_inches="tight", facecolor="#111111")
plt.show()
print("Saved: portfolio_outputs/portfolio_main.png", flush=True)

# =============================================
# DEEP DIVE — SCRATCH DEFECT
# =============================================
print("\nCreating scratch deep dive...", flush=True)

scratch_files = sorted(list((BASE / "scratch").glob("*.png")))

if scratch_files:
    img = cv2.imread(str(scratch_files[2]))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Anomalib saves images as side-by-side panels
    # Split into thirds: original | heatmap | ground truth
    w = img_rgb.shape[1]
    quarter = w // 4

    panel1 = img_rgb[:, :quarter, :]           # Original image
    panel2 = img_rgb[:, quarter*2:quarter*3, :] # Anomaly heatmap
    panel3 = img_rgb[:, quarter*3:, :]          # Prediction mask

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.patch.set_facecolor("#111111")
    fig2.suptitle(
        "Defect Localisation Deep Dive — Scratch Detection\n"
        "Verdict: DEFECTIVE COMPONENT — DO NOT SHIP",
        fontsize=13, fontweight="bold", color="#FF5722"
    )

    panel_titles = ["Original Component",
                    "Anomaly Heatmap", "Ground Truth Mask"]
    panel_colors = ["#4CAF50", "#FF9800", "#F44336"]

    for ax, panel, title, color in zip(
            axes2, [panel1, panel2, panel3],
            panel_titles, panel_colors):
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(title, color=color,
                     fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig("portfolio_outputs/defect_deep_dive.png",
                dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.show()
    print("Saved: portfolio_outputs/defect_deep_dive.png", flush=True)

print("\n" + "="*50, flush=True)
print("All done. Check portfolio_outputs/ folder.", flush=True)
print("="*50, flush=True)