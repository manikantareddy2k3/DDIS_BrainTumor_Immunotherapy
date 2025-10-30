# Data-Driven Immunotherapy Suitability (DDIS) for Brain Tumor Using Deep Learning

## Introduction
This project implements a **Data-Driven Immunotherapy Suitability (DDIS)** framework that leverages deep learning models to assist in assessing immunotherapy suitability for brain tumor patients.  
The framework integrates **U-Net** for tumor segmentation, **InceptionV3** for feature extraction and classification, and **Statistical Analysis** for evaluating model performance and biomarker relationships.

---

## Dataset Description

The dataset used in this project is sourced from **Figshare**:  
[Brain Tumor Dataset on Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5)  

This dataset includes **3,064 T1-weighted contrast-enhanced MRI images** collected from **233 patients** diagnosed with three types of brain tumors:

| Tumor Type       | Slice Count |
|------------------|-------------|
| Meningioma       | 708         |
| Glioma           | 1426        |
| Pituitary Tumor  | 930         |

Each image is stored in **MATLAB (.mat)** format, which includes structured data such as image arrays, tumor labels, and binary tumor masks.

###  Data Structure
Each `.mat` file contains the following fields:
- **cjdata.label:** Tumor type → `1 = Meningioma`, `2 = Glioma`, `3 = Pituitary Tumor`  
- **cjdata.PID:** Patient ID  
- **cjdata.image:** MRI image data  
- **cjdata.tumorBorder:** Vector of coordinates marking the tumor boundary (`[x1, y1, x2, y2, …]`)  
- **cjdata.tumorMask:** Binary image where `1` represents the tumor region  

The dataset is split into **four .zip files**, each containing ~766 MRI slices due to file size limits.  
5-fold cross-validation indices are also provided for experimental consistency.

###  Reference Papers
1. Cheng, Jun, et al. *"Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition."* PLOS One 10.10 (2015).  
2. Cheng, Jun, et al. *"Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation."* PLOS One 11.6 (2016).  
 [Original MATLAB Source Code (GitHub)](https://github.com/chengjun583/brainTumorRetrieval)

---

## Code Information

### 1️⃣ `unet_model.py` — Tumor Segmentation
- Implements a **U-Net** architecture for MRI-based brain tumor segmentation.  
- Preprocesses images (resized to 128×128, normalized) and generates binary masks.  
- Uses `binary_crossentropy` loss and `Adam` optimizer.  
- Outputs segmented tumor masks.  
- Model saved as `unet_brain_segmentation.h5`.

### 2️⃣ `inceptionv3_model.py` — Feature Extraction & Classification
- Utilizes **InceptionV3** (pretrained on ImageNet) for **feature extraction** and **tumor type classification**.  
- Classifies MRI scans into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.  
- Evaluates using accuracy, loss curves, and confusion matrix.  
- Model saved as `inceptionv3_brain_classification.h5`.

### 3️⃣ `statistical_analysis.py` — Statistical Evaluation
- Performs **statistical and performance analysis** on classification outputs.  
- Calculates metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
- Generates performance plots (confusion matrix, ROC curve).  

### 4️⃣ `final_combined.py` — Integrated DDIS Pipeline
- Integrates **U-Net** and **InceptionV3** into one pipeline:
  1. Load MRI image  
  2. Segment tumor (U-Net)  
  3. Classify tumor (InceptionV3)  
  4. Perform statistical evaluation  
- Provides end-to-end automated analysis.

---

##  Project Structure


---

##  Materials & Methods

###  Computing Infrastructure
- **Operating System:** Tested on **Ubuntu 20.04 (Kaggle environment)** and compatible with **Windows 10**.  
- **Hardware:**  
  - 16 GB RAM, Intel i7 CPU  
  - NVIDIA GPU (CUDA-compatible) recommended for TensorFlow training  
- **Software:**  
  - Python 3.10  
  - TensorFlow 2.13.0, NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn  
- **Dataset Location:**  
  - [Brain Tumor Dataset — Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5)

---

##  Evaluation Method

The proposed framework was evaluated using **train-test split** and **cross-validation**.

###  Evaluation Setup
- Dataset split: **80% training** and **20% testing**
- Models trained with Adam optimizer, learning rate = 1e-4
- Early stopping and model checkpoints used for stable convergence

###  Metrics Used
**Primary Metrics**
- Accuracy  
- Area Under ROC Curve (AUC)

**Secondary Metrics**
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

All metrics were calculated using `sklearn.metrics`, and performance plots (ROC curve, Precision-Recall curve) were generated for visualization.

---

## Limitations
- **Dataset Bias:** Limited patient diversity may restrict generalization to global populations.  
- **Preprocessing Loss:** Resizing may remove subtle tumor features.  
- **Transfer Learning Gap:** InceptionV3 pretrained on natural images may not fully adapt to MRI modalities.  
- **Compute Demand:** GPU acceleration required for efficient training; CPU-only systems are slower.  
- **Clinical Scope:** Current model focuses on classification, not survival or treatment response prediction.

---


