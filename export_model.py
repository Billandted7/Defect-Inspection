from pathlib import Path
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

print("Training model...", flush=True)

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="metal_nut",
    train_batch_size=16,
    eval_batch_size=1,
    num_workers=0,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

engine = Engine(default_root_dir="exported_model")
engine.fit(model=model, datamodule=datamodule)

print("Training complete. Exporting...", flush=True)

engine.export(
    model=model,
    export_type="torch",
    export_root=Path("exported_model"),
)

print("Export complete!", flush=True)
print("Check exported_model/ folder for output files.", flush=True)
