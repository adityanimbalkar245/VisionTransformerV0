# 🚀 VisionTransformerV0
ViT experiments on CIFAR-10 with supervised and DINO-pretrained models; includes training, patch size comparison, and saved weights.

This project explores the use of **Vision Transformers (ViT)** on the CIFAR-10 dataset under data and compute constraints. It benchmarks different ViT training strategies, patch sizes, and fine-tuning modes including full training and frozen-feature classification using DINO-pretrained backbones (ViT00–ViT02).

---

## 📂 Project Structure

| File/Folder                          | Description                                                              |
|-------------------------------------|--------------------------------------------------------------------------|
| `VIT00_Train_From_Scratch.ipynb`    | Trains ViT from scratch or with supervised pretraining on full model     |
| `VIT01_Patch_Analysis.ipynb`        | Trains ViT on **10% of CIFAR-10** with different patch sizes (16, 32)    |
| `VIT02_DINO_Frozen_Features.ipynb`  | Uses **DINO-pretrained ViT** as a frozen feature extractor + classifier  |
| `.pth files`                        | Saved model weights for each configuration                               |

---

## 📊 Key Results

| Model  | Patch Size | Training Strategy        | Accuracy | Training Time |
|--------|------------|--------------------------|----------|----------------|
| ViT00  | 16         | Full Fine-Tuning         | 93.78%   | ~158 min       |
| ViT01  | 16         | Fine-Tuning on 10%       | 84.0%    | ~18 min         |
| ViT01  | 32         | Fine-Tuning on 10%       | 81.0%    | ~14 min       |
| ViT02  | 16         | DINO + Frozen Encoder    | 91.9%    | ~3.5 min       |

---

## 💡 Highlights

- Modular PyTorch ViT implementation with CLS token, patch embedding, and attention blocks
- Experiments on low-data regimes with variable patch sizes
- Uses DINO-pretrained ViT for efficient feature extraction + lightweight classification
- Saved weights for reproducibility and comparison

---

## 📦 Requirements

- torch>=2.0
- torchvision
- timm
- matplotlib
- tqdm

---


