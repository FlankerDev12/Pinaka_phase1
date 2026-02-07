# Edge-AI Semiconductor Wafer Defect Classification

## Problem Statement
Real-time classification of semiconductor wafer defect patterns under edge compute constraints using lightweight deep learning models.

## Dataset
- Source: Public semiconductor wafer defect dataset (Kaggle)
- Total images: ~7,000+
- Classes (9):
  - Center
  - Donut
  - Edge-Loc
  - Edge-Ring
  - Local
  - Near-Full
  - Random
  - Scratch
  - Normal (Clean)
- Data split:
  - Train: 80%
  - Validation: 10%
  - Test: 10%
- Image type: Grayscale-equivalent (single channel after preprocessing)

Dataset link:  
👉 https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset

## Model
- Architecture: MobileNetV2
- Training approach: Transfer Learning
- Input size: 224 × 224
- Framework: PyTorch
- Model size: ~8.9 MB (ONNX)
- Target deployment: NXP eIQ on i.MX RT series devices

## Performance (Phase-1 Baseline)
- Accuracy: ~83–84%
- Precision: ~0.84
- Recall: ~0.82

Confusion matrix is available in the repository.

## Artifacts
- ONNX model:
  - `model/wafer_defect_mobilenetv2.onnx`
  - `model/wafer_defect_mobilenetv2.onnx.data`

## Notes
This repository contains Phase-1 baseline results. Further optimization, quantization, and deployment using NXP eIQ tools are planned in subsequent phases.
