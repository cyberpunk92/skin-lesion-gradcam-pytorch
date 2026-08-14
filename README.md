# Explainable Skin Lesion Classification with PyTorch and Grad-CAM

A medical computer vision and interpretability pipeline built with PyTorch. The system classifies dermoscopic skin lesion images and generates **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizations to highlight the discriminative morphological regions influencing model predictions.

---

## Table of Contents

- [Overview](#overview)
- [Explainability Results](#explainability-results)
- [Methodology and Architectural Details](#methodology-and-architectural-details)
- [Technical Stack](#technical-stack)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [License](#license)

---

## Overview

Deep learning models deployed in clinical dermatology must offer both high diagnostic sensitivity and interpretable decision-making. Standard black-box classification pipelines lack the transparency required for clinical validation.

This repository implements a modular explainable AI (XAI) workflow designed to:
1. Process and normalize dermoscopic lesion imagery according to standard ImageNet distributions.
2. Perform multi-class classification using an EfficientNet-B0 backbone.
3. Compute spatial gradients via backward hooks targeting the final convolutional layer.
4. Generate coarse localization heatmaps overlaid onto raw scans using a Jet colormap and alpha-blending.

---

## Explainability Results

Below is the side-by-side diagnostic visualization showing the original dermoscopic image alongside the overlaid Grad-CAM activation map:

![Grad-CAM Output](sample_gradcam_result.png)

---

## Methodology and Architectural Details

### 1. Feature Extraction Backbone
* **Model:** EfficientNet-B0 pre-trained on ImageNet.
* **Target Layer:** Final convolutional feature extractor (`features[-1]`).
* **Classification Head:** Linear projection adapted for multi-class dermatological categorization.

### 2. Gradient-Weighted Class Activation Mapping
* Computes the gradient of the predicted class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer.
* Calculates neuron importance weights $\alpha_k^c$ via global average pooling across spatial dimensions $(u, v)$:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
* Computes a weighted combination followed by a ReLU non-linearity to capture positively contributing spatial features:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

---

## Technical Stack

* **Deep Learning Framework:** PyTorch, Torchvision
* **Interpretability & Explainability:** pytorch-grad-cam
* **Image Processing & Augmentation:** OpenCV, Pillow, NumPy
* **Data Visualization:** Matplotlib

---

## Repository Structure

```text
skin-lesion-gradcam-pytorch/
├── main.py                     # Feature extraction, inference, and Grad-CAM generation
├── requirements.txt            # Project dependencies
├── sample_gradcam_result.png   # Diagnostic explanation figure
├── .gitignore                  # Ignored cache and environment directories
└── README.md                   # Project documentation
```

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/cyberpunk92/skin-lesion-gradcam-pytorch.git](https://github.com/cyberpunk92/skin-lesion-gradcam-pytorch.git)
cd skin-lesion-gradcam-pytorch
```

### 2. Create and Activate a Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the primary script to execute classification inference and export the side-by-side Grad-CAM diagnostic map:

```bash
python main.py
```

Upon execution, the script produces:
* `sample_gradcam_result.png`: High-resolution comparative visualization highlighting high-activation decision boundaries.

---

## License

This project is open-source and available under the MIT License.
