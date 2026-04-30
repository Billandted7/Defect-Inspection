import os
import json
import time
from pathlib import Path
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# =============================================
# EXPERIMENT CONFIGURATION
# We will test different backbones and 
# coreset sampling ratios and compare results
# =============================================

DATASET_PATH = Path("data/mvtec")
OUTPUT_DIR = Path("experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the experiments we want to run
# Each is a dictionary of settings to try
EXPERIMENTS = [
    {
        "name": "baseline_wide_resnet50",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "category": "metal_nut",
    },
    {
        "name": "resnet18_faster",
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "category": "metal_nut",
    },
    {
        "name": "larger_coreset",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.25,
        "num_neighbors": 9,
        "category": "metal_nut",
    },
    {
        "name": "more_neighbors",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 25,
        "category": "metal_nut",
    },
]

# =============================================
# RUN ALL EXPERIMENTS
# =============================================

all_results = []

for i, experiment in enumerate(EXPERIMENTS):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {experiment['name']}")
    print(f"Backbone: {experiment['backbone']}")
    print(f"Coreset ratio: {experiment['coreset_sampling_ratio']}")
    print(f"Neighbours: {experiment['num_neighbors']}")
    print("=" * 60)

    # Time how long each experiment takes
    start_time = time.time()

    # Set up data
    datamodule = MVTecAD(
        root=DATASET_PATH,
        category=experiment["category"],
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )

    # Set up model with this experiment's settings
    model = Patchcore(
        backbone=experiment["backbone"],
        layers=experiment["layers"],
        coreset_sampling_ratio=experiment["coreset_sampling_ratio"],
        num_neighbors=experiment["num_neighbors"],
    )

    # Set up engine with unique output dir per experiment
    experiment_output = OUTPUT_DIR / experiment["name"]
    engine = Engine(
        default_root_dir=str(experiment_output),
    )

    # Train
    print("\nTraining...")
    engine.fit(model=model, datamodule=datamodule)

    # Test
    print("Testing...")
    test_results = engine.test(model=model, datamodule=datamodule)

    elapsed = time.time() - start_time

    # Extract the metrics from results
    # test_results is a list of dicts
    metrics = test_results[0] if test_results else {}

    result_entry = {
        "name": experiment["name"],
        "backbone": experiment["backbone"],
        "coreset_sampling_ratio": experiment["coreset_sampling_ratio"],
        "num_neighbors": experiment["num_neighbors"],
        "image_AUROC": round(metrics.get("image_AUROC", 0), 4),
        "pixel_AUROC": round(metrics.get("pixel_AUROC", 0), 4),
        "image_F1Score": round(metrics.get("image_F1Score", 0), 4),
        "training_time_seconds": round(elapsed, 1),
    }

    all_results.append(result_entry)

    print(f"\nResult: image_AUROC={result_entry['image_AUROC']}, "
          f"pixel_AUROC={result_entry['pixel_AUROC']}, "
          f"time={result_entry['training_time_seconds']}s")

# =============================================
# SAVE AND DISPLAY RESULTS TABLE
# =============================================

# Save to JSON for later use
results_path = OUTPUT_DIR / "experiment_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60)
print("ALL EXPERIMENT RESULTS")
print("=" * 60)
print(f"{'Experiment':<25} {'image_AUROC':<14} {'pixel_AUROC':<14} {'Time(s)':<10}")
print("-" * 65)
for r in all_results:
    print(f"{r['name']:<25} {r['image_AUROC']:<14} {r['pixel_AUROC']:<14} {r['training_time_seconds']:<10}")

print(f"\nFull results saved to: {results_path}")
