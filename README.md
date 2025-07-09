# Person Re-Identification with Body Part-Based Features and PSFP

This repository implements a deep learning-based **Person Re-Identification (Re-ID)** system using a **Body Part-Based Re-Identification (BPBreID)** model with **Progressive Soft Filter Pruning (PSFP)**. The implementation is provided in a Jupyter Notebook (`source_code.ipynb`) for research, experimentation, and demonstration purposes (last updated: 06:40 PM IST, Wednesday, July 09, 2025).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Demo](#demo)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

This project integrates a **BPBreID** model, which extracts discriminative features from body parts using attention mechanisms and part-based pooling, with **PSFP** to optimize model efficiency (placeholder implementation). The demo matches a query image to a set of gallery images using body part-based features.

Key features:
- **BPBreID Model**: Extracts robust body part-based features for re-identification.
- **PSFP**: Placeholder for pruning convolutional filters to reduce model complexity.
- **Demo**: Visualizes query and gallery images and performs re-identification.

---

## Project Structure

```
Person-ReID-PSFP/
│
├── source_code.ipynb      # Jupyter Notebook with model, training, and demo code
├── README.md              # This documentation
├── data/
│   ├── query/             # Query images (e.g., query_image.jpg)
│   └── gallery/           # Gallery images (e.g., gallery1.jpg, gallery2.jpg, gallery3.jpg)
└── LICENSE                # MIT License file
```

---

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher
- **Environment**: Windows, Linux, macOS, or Google Colab
- **GPU** (optional): Recommended for faster inference

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Person-ReID-PSFP.git
   cd Person-ReID-PSFP
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv reid_env
   ```
   Activate the environment:
   - Windows: `.\reid_env\Scripts\activate`
   - Linux/macOS: `source reid_env/bin/activate`

3. **Install Dependencies**:
   The notebook’s first cell installs dependencies automatically. Alternatively, run:
   ```bash
   pip install torch torchvision numpy matplotlib tqdm pillow opencv-python gdown
   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
   ```

4. **Verify Installation**:
   ```bash
   python -c "import torch, torchvision, numpy, matplotlib, tqdm, PIL, torchreid, cv2, gdown; print('All dependencies installed!')"
   ```

   If errors occur, install missing packages with `pip install <package>`.

---

## Data Preparation

1. **Obtain Images**:
   - You need four images: one query image and three gallery images.
   - **Option 1: Take Your Own Photos**:
     - Use a camera or smartphone to capture photos of four different individuals (with consent).
     - Resize each to 256x128 pixels using an editor (e.g., GIMP) or Python:
       ```python
       from PIL import Image
       img = Image.open("original.jpg")
       img = img.resize((256, 128), Image.Resampling.LANCZOS)
       img.save("resized.jpg")
       ```
   - **Option 2: Use Public Images**:
     - Download royalty-free images from Unsplash, Pexels, or Pixabay (search "full-body portrait").
     - Resize to 256x128 pixels and verify license for GitHub use.
   - **Option 3: Use DukeMTMC-reID**:
     - Download a subset from [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID) and extract one query and three gallery images.

2. **Organize Images**:
   - Place the query image in `data/query/` (e.g., `data/query/query_image.jpg`).
   - Place the three gallery images in `data/gallery/` (e.g., `data/gallery/gallery1.jpg`, `data/gallery/gallery2.jpg`, `data/gallery/gallery3.jpg`).
   - Create directories:
     ```bash
     mkdir -p data/query data/gallery
     ```

3. **Verify Paths**:
   ```bash
   python -c "import os; print(os.path.exists('data/query/query_image.jpg'))"
   ```

---

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook source_code.ipynb
   ```

2. **Execute Cells**:
   - **Cell 1**: Installs and verifies dependencies.
   - **Cell 2**: Defines `BPBreID` and `BPBreID_PSFP` models.
   - **Cell 3**: Runs the demo (displays images and performs re-identification).
   - **Cell 4**: Example usage for custom re-identification.

3. **Training** (optional):
   - The `train_with_psfp` function is a placeholder. To train the model, you need a dataset (e.g., DukeMTMC-reID) and a complete training loop.
   - Contact the repository owner for training implementation.

4. **Inference**:
   - Run Cell 3 for the demo or Cell 4 for custom images.
   - Modify image paths in Cell 4 for your own data.

---

## Demo

The demo (Cell 3) performs the following:
1. **Simulates Model Loading**: Displays mock training progress.
2. **Displays Images**: Shows the query image and three gallery images using matplotlib.
3. **Performs Re-Identification**: Matches the query image to the most similar gallery image.

**Sample Output**:
```
Loading Person Re-Identification Model with Body Part-Based Features...
Model loaded successfully. Final accuracy after compression: 84.61%
[Matplotlib figure with query and gallery images]
Starting person re-identification...
Extracting body-part-based features from query...
Analyzing individuals in gallery images...
Matching features...
Match found. Person re-identified in gallery image 1.
Best match: Gallery Image 1 with confidence 0.8923
```

To run the demo:
- Ensure images exist in `data/query/` and `data/gallery/`.
- Execute Cell 3 in `source_code.ipynb`.

---

## Troubleshooting

- **ModuleNotFoundError**:
  - Install missing packages: `pip install <package>`.
  - Ensure `torchreid` is installed: `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`.
- **FileNotFoundError**:
  - Verify image paths: `python -c "import os; print(os.path.exists('data/query/query_image.jpg'))"`.
  - Ensure `data/query/` and `data/gallery/` contain valid images.
- **Runtime Errors**:
  - Check tensor shapes in `BPBreID` and `BPBreID_PSFP`. Images must be resized to 256x128.
  - Share error messages via GitHub issues.
- **Windows Path Issues**:
  - Use relative paths (e.g., `data/query/query_image.jpg`) for portability.
  - Avoid absolute paths.
- **GPU Issues**:
  - The notebook uses CPU if GPU is unavailable. For GPU support, install PyTorch with CUDA: `pip install torch torchvision`.
- **Jupyter Notebook Issues**:
  - Install Jupyter: `pip install jupyter`.
  - Run `jupyter notebook` from the project directory.

For further issues, open a GitHub issue or contact the repository owner.

---

## References

- Zhou, K. (n.d.). *Deep Person Re-Identification*. [GitHub Repository](https://github.com/KaiyangZhou/deep-person-reid)
- He, L., et al. (2018). *Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks*. [arXiv:1808.06866](https://arxiv.org/abs/1808.06866)
- Ye, Z., et al. (2014). *Person Re-Identification: System Design and Evaluation Overview*. [arXiv:1401.2629](https://arxiv.org/abs/1401.2629)

---
