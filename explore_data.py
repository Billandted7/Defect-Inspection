import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# CONFIGURATION - change these paths if needed
# =============================================
DATA_PATH = "data/mvtec/metal_nut"

# =============================================
# STEP 1: Count how many images we have
# =============================================
print("=" * 50)
print("DATASET OVERVIEW: METAL NUT")
print("=" * 50)

# Count training images
train_path = os.path.join(DATA_PATH, "train", "good")
train_images = os.listdir(train_path)
print(f"Training images (good only): {len(train_images)}")

# Count test images per defect type
test_path = os.path.join(DATA_PATH, "test")
test_categories = os.listdir(test_path)

print("\nTest images by category:")
total_test = 0
for category in test_categories:
    category_path = os.path.join(test_path, category)
    images = os.listdir(category_path)
    print(f"  {category}: {len(images)} images")
    total_test += len(images)

print(f"\nTotal test images: {total_test}")

# =============================================
# STEP 2: Check image dimensions
# =============================================
sample_image_path = os.path.join(train_path, train_images[0])
sample_image = cv2.imread(sample_image_path)
height, width, channels = sample_image.shape
print(f"\nImage dimensions: {width} x {height} pixels, {channels} colour channels")

# =============================================
# STEP 3: Visualise good vs defective images
# =============================================
print("\nGenerating visual comparison...")

# We want to show: 3 good images, then 1 of each defect type
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Metal Nut Dataset - Good vs Defective Parts", fontsize=16, fontweight='bold')

# Load and show 3 good training images
for i in range(3):
    img_path = os.path.join(train_path, train_images[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert colour format for display
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"GOOD (train)", color='green', fontweight='bold')
    axes[0, i].axis('off')

# Leave the 4th spot in row 1 blank
axes[0, 3].axis('off')

# Load and show one image of each defect type
defect_types = [c for c in test_categories if c != 'good']
for i, defect in enumerate(defect_types[:4]):
    defect_path = os.path.join(test_path, defect)
    defect_images = os.listdir(defect_path)
    img_path = os.path.join(defect_path, defect_images[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"DEFECT: {defect}", color='red', fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("data_exploration.png", dpi=150, bbox_inches='tight')
plt.show()
print("Visual saved as data_exploration.png")

# =============================================
# STEP 4: Look at a defect with its ground truth mask
# =============================================
print("\nGenerating defect mask comparison...")

# Pick a scratched nut and show it alongside its ground truth mask
scratch_test_path = os.path.join(DATA_PATH, "test", "scratch")
scratch_mask_path = os.path.join(DATA_PATH, "ground_truth", "scratch")

scratch_images = sorted(os.listdir(scratch_test_path))
scratch_masks = sorted(os.listdir(scratch_mask_path))

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
fig2.suptitle("Defect Localisation - Where Is The Scratch?", fontsize=14, fontweight='bold')

# Original good image for reference
good_img_path = os.path.join(train_path, train_images[0])
good_img = cv2.imread(good_img_path)
good_img = cv2.cvtColor(good_img, cv2.COLOR_BGR2RGB)
axes2[0].imshow(good_img)
axes2[0].set_title("Good Part (reference)", color='green')
axes2[0].axis('off')

# Defective image
defect_img_path = os.path.join(scratch_test_path, scratch_images[0])
defect_img = cv2.imread(defect_img_path)
defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB)
axes2[1].imshow(defect_img)
axes2[1].set_title("Defective Part (scratch)", color='red')
axes2[1].axis('off')

# Ground truth mask - white pixels = defect location
mask_path = os.path.join(scratch_mask_path, scratch_masks[0])
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
axes2[2].imshow(mask, cmap='hot')
axes2[2].set_title("Ground Truth Mask\n(white = defect location)", color='orange')
axes2[2].axis('off')

plt.tight_layout()
plt.savefig("defect_mask_example.png", dpi=150, bbox_inches='tight')
plt.show()
print("Mask comparison saved as defect_mask_example.png")

print("\n" + "=" * 50)
print("Data exploration complete.")
print("=" * 50)