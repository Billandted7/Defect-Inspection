"""
PatchCore Anomaly Detection — Built From Scratch
=================================================
Implements the core PatchCore algorithm without
using the Anomalib library. Uses only:
- PyTorch (feature extraction)
- torchvision (pretrained ResNet50)
- NumPy (coreset sampling, distance calculation)
- scikit-learn (nearest neighbour search)
- OpenCV (image processing)
- Matplotlib (visualisation)

Reference:
Roth et al., "Towards Total Recall in Industrial
Anomaly Detection", CVPR 2022.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# =============================================
# COMPONENT 1: FEATURE EXTRACTOR
# =============================================

class FeatureExtractor(nn.Module):
    """
    Wraps a pretrained ResNet50 and extracts
    intermediate layer features using hooks.

    We use layer2 and layer3 because:
    - layer2 captures fine details (textures,
      small scratches)
    - layer3 captures broader structure (shape,
      large deformations)
    - Earlier layers are too low-level
    - Later layers are too abstract/semantic
    """

    def __init__(self):
        super().__init__()

        print("Loading pretrained ResNet50...",
              flush=True)

        # Load ResNet50 pretrained on ImageNet
        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights
            .IMAGENET1K_V1
        )

        # We only need layers up to layer3
        # Remove the classification head
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Freeze all weights — we are only using
        # this network as a feature extractor,
        # not training it
        for param in self.parameters():
            param.requires_grad = False

        self.eval()
        print("Feature extractor ready.", flush=True)

    def forward(self, x):
        """
        Forward pass returns features from
        layer2 and layer3.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        layer2_features = self.layer2(x)
        layer3_features = self.layer3(layer2_features)
        return layer2_features, layer3_features


# =============================================
# COMPONENT 2: PATCH FEATURE BUILDER
# =============================================

def extract_patch_features(feature_extractor,
                            img_tensor):
    """
    Extract and combine patch features from
    layer2 and layer3 for a single image.

    Feature maps are spatial grids — each cell
    in the grid corresponds to a patch of the
    original image. We:
    1. Extract features from both layers
    2. Upsample layer3 to match layer2 spatial size
    3. Concatenate along the feature dimension
    4. Apply average pooling to aggregate
       neighbouring patches (reduces noise)
    5. Reshape to (num_patches, feature_dim)

    Returns:
        numpy array of shape
        (num_patches, feature_dim)
    """
    with torch.no_grad():
        layer2_feat, layer3_feat = \
            feature_extractor(img_tensor)

    # layer2: (1, 512, 32, 32) for 256x256 input
    # layer3: (1, 1024, 16, 16) for 256x256 input

    # Upsample layer3 to match layer2 spatial size
    layer3_upsampled = nn.functional.interpolate(
        layer3_feat,
        size=layer2_feat.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # Concatenate: (1, 1536, 32, 32)
    combined = torch.cat(
        [layer2_feat, layer3_upsampled], dim=1)

    # Adaptive average pooling with kernel 3
    # Aggregates each patch with its neighbours
    # This makes features more robust to small
    # misalignments
    pooled = nn.functional.avg_pool2d(
        combined,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Reshape to (num_patches, feature_dim)
    # Permute: (1, feat_dim, H, W) ->
    #          (1, H, W, feat_dim)
    # Reshape: (H*W, feat_dim)
    b, c, h, w = pooled.shape
    patches = pooled.permute(0, 2, 3, 1)
    patches = patches.reshape(-1, c)
    patches = patches.numpy()

    return patches, (h, w)


# =============================================
# COMPONENT 3: CORESET SAMPLER
# =============================================

def coreset_sampling(features, ratio=0.1):
    """
    Greedy coreset sampling to select a
    representative subset of feature vectors.

    Algorithm:
    1. Start with a random point
    2. Compute distance from all points to the
       nearest already-selected point
    3. Select the point with maximum minimum
       distance (most different from all selected)
    4. Repeat until we have ratio * N points

    This ensures the selected subset covers
    the full distribution without redundancy.

    Args:
        features: numpy array (N, feature_dim)
        ratio: fraction of points to keep

    Returns:
        numpy array of selected features
    """
    n_total = len(features)
    n_select = max(1, int(n_total * ratio))

    print(f"  Coreset sampling: {n_total} -> "
          f"{n_select} patches ({ratio*100:.0f}%)",
          flush=True)

    # Start with a random point
    selected_indices = [
        np.random.randint(0, n_total)]

    # Track minimum distance from each point
    # to the nearest selected point
    # Initialise with distance to first selected
    selected_feat = features[
        selected_indices[0]:selected_indices[0]+1]

    # Compute initial distances
    # Using squared Euclidean for speed
    min_distances = np.sum(
        (features - selected_feat) ** 2, axis=1)

    for i in range(n_select - 1):
        # Select point with maximum min distance
        new_idx = int(np.argmax(min_distances))
        selected_indices.append(new_idx)

        # Update minimum distances
        new_feat = features[new_idx:new_idx+1]
        new_distances = np.sum(
            (features - new_feat) ** 2, axis=1)
        min_distances = np.minimum(
            min_distances, new_distances)

        # Progress every 10%
        if (i + 1) % max(
                1, (n_select // 10)) == 0:
            pct = (i + 1) / n_select * 100
            print(f"  Coreset progress: "
                  f"{pct:.0f}%", flush=True)

    coreset = features[selected_indices]
    print(f"  Coreset complete: "
          f"{len(coreset)} patches selected",
          flush=True)

    return coreset


# =============================================
# COMPONENT 4: ANOMALY SCORER
# =============================================

class PatchCoreModel:
    """
    Complete PatchCore model combining all
    four components.

    Attributes:
        feature_extractor: pretrained ResNet50
        memory_bank: coreset of training patches
        knn: fitted nearest neighbour model
        spatial_shape: (H, W) of feature maps
    """

    def __init__(self,
                 coreset_ratio=0.1,
                 n_neighbors=9):
        self.feature_extractor = FeatureExtractor()
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.memory_bank = None
        self.knn = None
        self.spatial_shape = None

        # ImageNet normalisation — must match
        # what the pretrained ResNet expects
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def preprocess(self, img_rgb):
        """Convert RGB image to normalised tensor."""
        tensor = self.transform(img_rgb)
        return tensor.unsqueeze(0)  # Add batch dim

    def fit(self, image_paths):
        """
        Build memory bank from training images.

        For each training image:
        1. Preprocess and extract patch features
        2. Accumulate all patches
        Then apply coreset sampling to reduce size.
        Finally fit KNN on the coreset.
        """
        print(f"\nBuilding memory bank from "
              f"{len(image_paths)} training images...",
              flush=True)

        all_patches = []

        for i, img_path in enumerate(
                tqdm(image_paths,
                     desc="Extracting features")):
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB)

            # Preprocess
            img_tensor = self.preprocess(img_rgb)

            # Extract patch features
            patches, spatial_shape = \
                extract_patch_features(
                    self.feature_extractor,
                    img_tensor
                )

            all_patches.append(patches)

            # Store spatial shape from first image
            if self.spatial_shape is None:
                self.spatial_shape = spatial_shape

        # Stack all patches: (N_total, feature_dim)
        all_patches = np.vstack(all_patches)
        print(f"\nTotal patches collected: "
              f"{len(all_patches):,}", flush=True)
        print(f"Feature dimension: "
              f"{all_patches.shape[1]}", flush=True)

        # Apply coreset sampling
        print("\nApplying coreset sampling...",
              flush=True)
        self.memory_bank = coreset_sampling(
            all_patches, self.coreset_ratio)

        # Fit KNN on memory bank
        print("\nFitting KNN on memory bank...",
              flush=True)
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1
        )
        self.knn.fit(self.memory_bank)
        print(f"KNN fitted on "
              f"{len(self.memory_bank):,} "
              f"memory bank patches.", flush=True)

    def predict(self, img_rgb):
        """
        Compute anomaly score and heatmap for
        a single test image.

        Process:
        1. Extract patch features from test image
        2. For each patch, find k nearest neighbours
           in memory bank
        3. Anomaly score = mean distance to
           k nearest neighbours
        4. Image score = max patch score
           (most anomalous patch determines result)
        5. Reshape patch scores to 2D heatmap

        Returns:
            anomaly_score: float (image-level)
            anomaly_map: 2D numpy array (heatmap)
            img_resized: resized input image
        """
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_tensor = self.preprocess(img_resized)

        # Extract patches
        patches, _ = extract_patch_features(
            self.feature_extractor, img_tensor)

        # KNN distance for each patch
        distances, _ = self.knn.kneighbors(patches)

        # Mean distance across k neighbours
        patch_scores = distances.mean(axis=1)

        # Image-level score = max patch score
        anomaly_score = float(patch_scores.max())

        # Reshape to spatial heatmap
        h, w = self.spatial_shape
        anomaly_map = patch_scores.reshape(h, w)

        # Upsample heatmap to image size
        anomaly_map = cv2.resize(
            anomaly_map, (256, 256),
            interpolation=cv2.INTER_LINEAR)

        # Normalise to 0-1
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (
                (anomaly_map - anomaly_map.min()) /
                (anomaly_map.max() -
                 anomaly_map.min())
            )

        return anomaly_score, anomaly_map, img_resized

    def save(self, path):
        """Save memory bank and KNN to disk."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "memory_bank.npy",
                self.memory_bank)
        joblib.dump(self.knn, path / "knn.joblib")
        np.save(path / "spatial_shape.npy",
                np.array(self.spatial_shape))
        print(f"Model saved to {path}", flush=True)

    def load(self, path):
        """Load memory bank and KNN from disk."""
        import joblib
        path = Path(path)
        self.memory_bank = np.load(
            path / "memory_bank.npy")
        self.knn = joblib.load(
            path / "knn.joblib")
        self.spatial_shape = tuple(
            np.load(path / "spatial_shape.npy"))
        print(f"Model loaded from {path}",
              flush=True)


# =============================================
# TRAINING
# =============================================

def train(dataset_path="data/mvtec",
          category="metal_nut",
          save_path="scratch_model"):
    """
    Train the from-scratch PatchCore model
    on the MVTec dataset.
    """
    print("=" * 60, flush=True)
    print("PATCHCORE FROM SCRATCH — TRAINING",
          flush=True)
    print("=" * 60, flush=True)

    # Find training images
    train_path = Path(dataset_path) / category \
        / "train" / "good"
    image_paths = sorted(
        list(train_path.glob("*.png")))

    print(f"Training images found: "
          f"{len(image_paths)}", flush=True)

    # Initialise and train model
    model = PatchCoreModel(
        coreset_ratio=0.1,
        n_neighbors=9
    )

    start = time.time()
    model.fit(image_paths)
    elapsed = time.time() - start

    print(f"\nTraining complete in "
          f"{elapsed:.0f} seconds.", flush=True)

    # Save model
    model.save(save_path)

    return model


# =============================================
# EVALUATION
# =============================================

def evaluate(model,
             dataset_path="data/mvtec",
             category="metal_nut"):
    """
    Evaluate the model on all test images.
    Computes Image AUROC and Pixel AUROC
    to compare with Anomalib baseline.
    """
    print("\n" + "=" * 60, flush=True)
    print("EVALUATING ON TEST SET", flush=True)
    print("=" * 60, flush=True)

    test_path = Path(dataset_path) / category \
        / "test"
    gt_path = Path(dataset_path) / category \
        / "ground_truth"

    all_scores = []
    all_labels = []
    all_maps = []
    all_gt_maps = []

    defect_types = sorted([
        d.name for d in test_path.iterdir()
        if d.is_dir()
    ])

    for defect_type in defect_types:
        is_good = defect_type == "good"
        label = 0 if is_good else 1

        images = sorted(list(
            (test_path / defect_type).glob("*.png")))

        print(f"\nTesting {defect_type}: "
              f"{len(images)} images", flush=True)

        for img_path in tqdm(
                images, desc=f"  {defect_type}"):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB)

            score, anomaly_map, _ = \
                model.predict(img_rgb)

            all_scores.append(score)
            all_labels.append(label)
            all_maps.append(anomaly_map.flatten())

            # Load ground truth mask
            if not is_good:
                mask_name = img_path.stem \
                    + "_mask.png"
                mask_path = (gt_path / defect_type
                             / mask_name)
                if mask_path.exists():
                    mask = cv2.imread(
                        str(mask_path),
                        cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(
                        mask, (256, 256))
                    mask_binary = (
                        mask > 0).astype(int)
                    all_gt_maps.append(
                        mask_binary.flatten())
                else:
                    all_gt_maps.append(
                        np.zeros(
                            256 * 256, dtype=int))
            else:
                all_gt_maps.append(
                    np.zeros(
                        256 * 256, dtype=int))

    # Compute AUROC scores
    image_auroc = roc_auc_score(
        all_labels, all_scores)

    pixel_auroc = roc_auc_score(
        np.concatenate(all_gt_maps),
        np.concatenate(all_maps)
    )

    print("\n" + "=" * 60, flush=True)
    print("RESULTS — FROM SCRATCH IMPLEMENTATION",
          flush=True)
    print("=" * 60, flush=True)
    print(f"Image AUROC: {image_auroc:.4f}",
          flush=True)
    print(f"Pixel AUROC: {pixel_auroc:.4f}",
          flush=True)
    print("\nAnomlib baseline for comparison:",
          flush=True)
    print("  Image AUROC: 0.9976", flush=True)
    print("  Pixel AUROC: 0.9868", flush=True)
    print("=" * 60, flush=True)

    return image_auroc, pixel_auroc, all_scores, \
        all_labels


# =============================================
# VISUALISATION
# =============================================

def visualise_results(model,
                      dataset_path="data/mvtec",
                      category="metal_nut"):
    """
    Generate visualisation showing good vs
    defective parts with heatmaps.
    """
    print("\nGenerating visualisations...",
          flush=True)

    os.makedirs("scratch_outputs", exist_ok=True)

    test_path = Path(dataset_path) / category \
        / "test"

    defect_types = ["good", "bent",
                    "color", "flip", "scratch"]

    results = []

    for defect_type in defect_types:
        img_path = sorted(list(
            (test_path / defect_type).glob(
                "*.png")))[0]

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB)
        score, anomaly_map, img_resized = \
            model.predict(img_rgb)
        results.append((
            defect_type, img_resized,
            anomaly_map, score))

    # Create figure
    n = len(results)
    fig, axes = plt.subplots(
        2, n, figsize=(5 * n, 10))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        "PatchCore From Scratch — "
        "Metal Nut Defect Detection",
        fontsize=14, fontweight="bold",
        color="white"
    )

    for i, (defect_type, img_resized,
            anomaly_map, score) in \
            enumerate(results):
        is_good = defect_type == "good"
        color = "#4CAF50" if is_good else "#FF5722"
        label = ("GOOD — PASS ✓"
                 if is_good
                 else f"FAIL: {defect_type.upper()}")

        # Top row: original image
        axes[0, i].imshow(img_resized)
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"{label}\nScore: {score:.3f}",
            color=color, fontsize=9,
            fontweight="bold"
        )

        # Bottom row: anomaly heatmap
        axes[1, i].imshow(
            anomaly_map, cmap="jet")
        axes[1, i].axis("off")
        axes[1, i].set_title(
            "Anomaly Map",
            color="#FF9800", fontsize=9
        )

    plt.tight_layout()
    output_path = ("scratch_outputs/"
                   "scratch_results.png")
    plt.savefig(output_path, dpi=150,
                bbox_inches="tight",
                facecolor="#111111")
    plt.show()
    print(f"Saved: {output_path}", flush=True)


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":

    # Install joblib if needed
    try:
        import joblib
    except ImportError:
        os.system("pip install joblib")
        import joblib

    scratch_model_path = "scratch_model"

    # Train if no saved model exists
    if not Path(
            scratch_model_path
    ).exists():
        model = train(
            save_path=scratch_model_path)
    else:
        print("Loading existing scratch model...",
              flush=True)
        model = PatchCoreModel()
        model.load(scratch_model_path)

    # Evaluate
    image_auroc, pixel_auroc, scores, labels = \
        evaluate(model)

    # Visualise
    visualise_results(model)

    print("\nPhase 7 complete.", flush=True)
    print("Check scratch_outputs/ for results.",
          flush=True)