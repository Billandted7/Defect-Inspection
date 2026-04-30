from pathlib import Path
import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode

# =============================================
# CONFIGURATION
# =============================================
DATASET_PATH = Path("data/mvtec")
CATEGORY = "metal_nut"
OUTPUT_DIR = Path("results")
IMAGE_SIZE = 256

print("=" * 60)
print("F1 COMPONENT DEFECT DETECTION SYSTEM")
print("Using PatchCore Anomaly Detection")
print("=" * 60)
print(f"Category: {CATEGORY}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# =============================================
# STEP 1: SET UP THE DATA
# =============================================
print("\n[1/3] Setting up dataset...")

datamodule = MVTecAD(
    root=DATASET_PATH,
    category=CATEGORY,
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=0,
)

print("Dataset ready")

# =============================================
# STEP 2: SET UP THE MODEL
# =============================================
print("\n[2/3] Setting up PatchCore model...")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

print("Model ready")

# =============================================
# STEP 3: TRAIN AND TEST
# =============================================
print("\n[3/3] Starting training...")
print("This will take 5-20 minutes on CPU. Do not close the window.")
print("-" * 60)

engine = Engine(
    default_root_dir=str(OUTPUT_DIR),
)

engine.fit(
    model=model,
    datamodule=datamodule,
)

print("\nTraining complete")

# =============================================
# STEP 4: TEST
# =============================================
print("\nEvaluating on test set...")

test_results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 60)
print("Key metric to look for: image_AUROC")
print("Score above 0.90 = strong performance")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 60)
