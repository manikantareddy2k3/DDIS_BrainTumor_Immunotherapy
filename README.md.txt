# 🧠 Data-Driven Immunotherapy Suitability (DDIS) for Brain Tumor Using Deep Learning

## Introduction
This project implements a **Data-Driven Immunotherapy Suitability (DDIS)** framework that leverages deep learning models to assist in assessing immunotherapy suitability for brain tumor patients.  
It uses **U-Net** for tumor segmentation, **InceptionV3** for feature extraction and classification, and **Statistical Analysis** for evaluating model performance and biomarker relationships.

---

## Dataset Description
The dataset consists of MRI scans of four brain tumor categories:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor (Control group)

Each MRI image is preprocessed (resized, normalized) for training and evaluation.  
The dataset is valuable for analyzing **tumor morphology**, **biomarker associations**, and **treatment suitability prediction**.

---

## Code Information

### 1️⃣ `unet_model.py` — Tumor Segmentation
- Implements a **U-Net** architecture for precise brain tumor segmentation.  
- Preprocesses MRI images, resizes to (128×128), and normalizes pixel values.  
- Generates **binary masks** representing tumor regions.  
- Trains with `binary_crossentropy` loss and `Adam` optimizer.  
- Saves the model as `unet_brain_segmentation.h5`.

### 2️⃣ `inceptionv3_model.py` — Feature Extraction & Classification
- Uses **InceptionV3** pretrained on ImageNet for **feature extraction** and **tumor classification**.  
- Extracts deep features from segmented images.  
- Classifies into four categories: Glioma, Meningioma, Pituitary, and No Tumor.  
- Evaluates accuracy, loss, and confusion matrix.  
- Saves model as `inceptionv3_brain_classification.h5`.

### 3️⃣ `statistical_analysis.py` — Statistical Evaluation
- Performs statistical analysis on classification results.  
- Computes metrics such as **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.  
- Generates summary statistics and correlation analysis of biomarkers (if available).

### 4️⃣ `final_combined.py` — Integrated DDIS Pipeline
- Integrates **U-Net segmentation** and **InceptionV3 classification**.  
- Loads pre-trained models, performs segmentation → classification → analysis sequentially.  
- Displays and saves final predictions and performance metrics.  
- Represents the complete DDIS workflow.

---

## 🧩 Project Structure
