## Related Research Article

This repository accompanies the manuscript:

**“Enhanced Spatiotemporal Degradation Modeling: A Robust Hybrid CNN–LSTM Approach”**


If you use this dataset, code, or benchmark framework in your research, please cite the corresponding article.

Zenodo Archive (Dataset & Code):
https://doi.org/10.5281/zenodo.18737669
# Robust Hybrid CNN–LSTM Framework for Spatiotemporal Surface Degradation Modeling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18737669.svg)](https://doi.org/10.5281/zenodo.18737669)

## Overview

This repository contains the official implementation of the paper:

**"Robust Hybrid CNN–LSTM Framework for Spatiotemporal Surface Degradation Modeling Under Controlled Temporal Perturbations."**

This work introduces:

- A controlled synthetic spatiotemporal corrosion benchmark
- Ambiguity-aware degradation modeling
- CNN vs CNN–LSTM comparative study
- Temporal jitter robustness evaluation
- Explainable AI analysis via Grad-CAM
- Full reproducibility via open-source release

---

## Dataset Description

The synthetic benchmark includes:

- 4 degradation severity classes  
- 5-frame temporal sequences  
- Progressive corrosion evolution  
- Ambiguity injection between adjacent severity levels  
- Controlled temporal jitter perturbation protocol  

### Dataset Statistics

- Total Images: ~5000+
- Classes: 4
- Sequence Length: 5 frames
- Train/Validation/Test Split: 70% / 15% / 15%
- Image Resolution: 224×224
- Augmentation: rotation, blur, noise injection

---

## Reproducibility

All dataset generators, model weights, training scripts, and evaluation protocols are publicly available.

Permanent archive:  
DOI: **10.5281/zenodo.18737669**

---
### Citation

If you use this dataset or code in your research, please cite:

@article{Arora2026_spatiotemporal_corrosion,
  author  = {Siddharth Arora},
  title   = {Enhanced Spatiotemporal Degradation Modeling: A Robust Hybrid CNN--LSTM Approach},
  year    = {2026},
  note    = {Research manuscript}
}
The code and dataset are released to support reproducible research in spatiotemporal surface degradation modeling.

## Repository Structure

dataset_generator/      Synthetic corrosion dataset generator

models/                 CNN and CNN–LSTM architecture implementation

training/               Training scripts and model optimization

evaluation/             Performance evaluation and robustness testing

visualization/          Grad-CAM visualization and interpretability tools

configs/                Experimental configuration files

## Installation

Create environment:

```bash
conda create -n corrosion_env python=3.10

## Quick Start (Reproducibility Guide)

This repository provides the dataset generator, training pipeline, and evaluation scripts for reproducing the experiments reported in the manuscript.

### 1. Clone the Repository
git clone https://github.com/siddhartharora10/corrosion-spatiotemporal-benchmark.git
cd corrosion-spatiotemporal-benchmark

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Generate the Synthetic Dataset
python dataset_generator.py

### 4. Train the CNN–LSTM Model
python train_hybrid_model.py

### 5. Evaluate the Model
python evaluate_model.py

### 6. Generate Grad-CAM Visualizations
python gradcam_visualization.py
conda activate corrosion_env
pip install -r requirements.txt
