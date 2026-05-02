import os
import cv2
import numpy as np
from pathlib import Path

os.environ["TRUST_REMOTE_CODE"] = "1"

from anomalib.deploy import TorchInferencer

print("Loading model...", flush=True)

inferencer = TorchInferencer(
    path=Path("exported_model/weights/torch/model.pt"),
)

# Run all good test images through
good_path = Path("data/mvtec/metal_nut/test/good")
good_images = sorted(list(good_path.glob("*.png")))

print(f"Testing {len(good_images)} good images...",
      flush=True)

good_scores = []
for img_path in good_images:
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = inferencer.predict(image=img_rgb)
    score = result.pred_score
    if hasattr(score, "numpy"):
        score = float(score.numpy().flatten()[0])
    else:
        score = float(score)
    good_scores.append(score)
    print(f"  {img_path.name}: {score:.4f}",
          flush=True)

# Now run all defective images
defect_scores = []
defect_types = ["bent", "color", "flip", "scratch"]

for defect in defect_types:
    defect_path = Path(
        f"data/mvtec/metal_nut/test/{defect}")
    images = sorted(list(defect_path.glob("*.png")))
    print(f"\nTesting {len(images)} {defect} images...",
          flush=True)
    for img_path in images:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = inferencer.predict(image=img_rgb)
        score = result.pred_score
        if hasattr(score, "numpy"):
            score = float(score.numpy().flatten()[0])
        else:
            score = float(score)
        defect_scores.append(score)

print("\n" + "=" * 50, flush=True)
print("CALIBRATION RESULTS", flush=True)
print("=" * 50, flush=True)
print(f"\nGood images ({len(good_scores)} total):",
      flush=True)
print(f"  Min score:  {min(good_scores):.4f}",
      flush=True)
print(f"  Max score:  {max(good_scores):.4f}",
      flush=True)
print(f"  Mean score: {np.mean(good_scores):.4f}",
      flush=True)
print(f"  95th percentile: "
      f"{np.percentile(good_scores, 95):.4f}",
      flush=True)
print(f"  99th percentile: "
      f"{np.percentile(good_scores, 99):.4f}",
      flush=True)

print(f"\nDefective images ({len(defect_scores)} total):",
      flush=True)
print(f"  Min score:  {min(defect_scores):.4f}",
      flush=True)
print(f"  Max score:  {max(defect_scores):.4f}",
      flush=True)
print(f"  Mean score: {np.mean(defect_scores):.4f}",
      flush=True)

# Recommended threshold
p95 = np.percentile(good_scores, 95)
p99 = np.percentile(good_scores, 99)
defect_min = min(defect_scores)

print(f"\nRECOMMENDED THRESHOLD: {p99:.4f}", flush=True)
print(f"(99th percentile of good scores)", flush=True)
print(f"\nAt this threshold:", flush=True)
good_pass = sum(s <= p99 for s in good_scores)
defect_fail = sum(s > p99 for s in defect_scores)
print(f"  Good parts passing: "
      f"{good_pass}/{len(good_scores)} "
      f"({good_pass/len(good_scores)*100:.0f}%)",
      flush=True)
print(f"  Defects caught: "
      f"{defect_fail}/{len(defect_scores)} "
      f"({defect_fail/len(defect_scores)*100:.0f}%)",
      flush=True)
print("=" * 50, flush=True)
