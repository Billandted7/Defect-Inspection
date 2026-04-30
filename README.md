# Computer Vision Defect Inspection System
### Automated Visual Quality Control for Safety-Critical Manufacturing Components

![Main Results](docs/portfolio_main.png)

---

## Project Overview

This project builds an automated visual inspection system capable of detecting 
surface defects on manufactured components using unsupervised deep learning. 
The system is motivated by quality engineering challenges in high-performance 
manufacturing environments such as Formula 1, where a single defective component 
reaching a race car can have catastrophic consequences.

The core problem: in real manufacturing you cannot collect thousands of defective 
parts to train a classifier. Instead, this system trains exclusively on defect-free 
components and learns to flag anything that deviates from normal — mirroring how 
real industrial inspection systems must operate.

---

## Results

| Metric | Score |
|--------|-------|
| Image AUROC | **0.9976** |
| Pixel AUROC | **0.9868** |
| Image F1 Score | **0.9838** |
| Pixel F1 Score | **0.8394** |

> Image AUROC measures how accurately the system classifies whole components 
> as good or defective. Pixel AUROC measures how precisely it localises the 
> defect location. Both exceed 0.98 — well above the 0.90 threshold considered 
> strong performance in academic literature.

---

## Defect Types Detected

The system was trained and evaluated on the MVTec Anomaly Detection dataset 
using the metal nut category, covering four defect types:

- **Bent** — physical deformation of the component
- **Colour** — surface contamination or staining  
- **Flip** — incorrect orientation
- **Scratch** — surface damage

![Dataset Overview](docs/data_exploration.png)

---

## How It Works

The system uses **PatchCore** — a state-of-the-art unsupervised anomaly detection 
algorithm. The approach:

1. A pretrained WideResNet50 neural network extracts patch-level features from 
   every defect-free training image
2. These features are stored in a memory bank representing what "normal" looks like
3. At inspection time, each patch of the test image is compared to its nearest 
   neighbours in the memory bank
4. Patches far from any normal example generate high anomaly scores
5. These scores are assembled into a heatmap showing exactly where the defect is

This approach requires **no defective training examples** — making it practical 
for real manufacturing where defects are rare.

![Defect Localisation](docs/defect_mask_example.png)

---

## Defect Localisation

The system outputs pixel-level anomaly maps highlighting precisely where defects 
are located — not just whether a part is defective, but where to look.

![Deep Dive](docs/defect_deep_dive.png)

---

## Experiments

Four experiments were run to evaluate how model architecture and hyperparameters 
affect performance and speed:

![Experiment Results](docs/experiment_comparison.png)

![Results Table](docs/results_table.png)

---

## Tech Stack

- **Python 3.14**
- **PyTorch** — deep learning framework
- **Anomalib 2.4.0** — anomaly detection library by Intel
- **OpenCV** — image processing
- **Matplotlib** — visualisation

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Billandted7/Defect-Inspection.git
cd Defect-Inspection
```

**2. Create a virtual environment and install dependencies**
```bash
py -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install anomalib opencv-python matplotlib scikit-learn pillow streamlit
```

**3. Download the dataset**

Download the MVTec Anomaly Detection dataset from 
[mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) 
and place it at `data/mvtec/metal_nut/`

**4. Train and evaluate**
```bash
python train_model.py
```

**5. Run experiments**
```bash
python experiments.py
```

**6. Generate visualisations**
```bash
python visualise_results.py
python visualise_predictions.py
```

---

## Quality Engineering Context

This project directly addresses real challenges in manufacturing quality control:

- **Right-First-Time manufacturing** — catching defects before components are 
  assembled reduces rework cost and schedule risk
- **Scalability** — automated inspection can process hundreds of parts per minute 
  versus manual inspection
- **Traceability** — every inspection result is logged with a timestamp, image 
  path, anomaly score and pass/fail verdict
- **No defective samples required** — the unsupervised approach works in real 
  factory conditions where defect data is scarce

---

Dataset: MVTec AD — Bergmann et al., CVPR 2019*