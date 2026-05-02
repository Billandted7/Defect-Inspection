import cv2
import numpy as np
from pathlib import Path
from anomalib.deploy import TorchInferencer

print("Loading inferencer...", flush=True)

inferencer = TorchInferencer(
    path=Path("exported_model/weights/torch/model.pt"),
)

print("Inferencer loaded.", flush=True)

# Test on a known scratch image
test_image_path = "data/mvtec/metal_nut/test/scratch/000.png"
img = cv2.imread(test_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Running inference on scratch image...", flush=True)
result = inferencer.predict(image=img_rgb)

print(f"Anomaly score: {result.pred_score}", flush=True)
print(f"Pred label: {result.pred_label}", flush=True)

# Test on a good image
good_image_path = "data/mvtec/metal_nut/test/good/000.png"
img2 = cv2.imread(good_image_path)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

print("\nRunning inference on good image...", flush=True)
result2 = inferencer.predict(image=img2_rgb)

print(f"Anomaly score: {result2.pred_score}", flush=True)
print(f"Pred label: {result2.pred_label}", flush=True)

print("\nDone.", flush=True)