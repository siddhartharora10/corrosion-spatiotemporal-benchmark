## Related Research Article

This repository accompanies the manuscript:

**“Enhanced Spatiotemporal Degradation Modeling: A Robust Hybrid CNN–LSTM Approach”**

submitted to *The Visual Computer*.

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

## Installation

Create environment:

```bash
conda create -n corrosion_env python=3.10
conda activate corrosion_env
pip install -r requirements.txt
