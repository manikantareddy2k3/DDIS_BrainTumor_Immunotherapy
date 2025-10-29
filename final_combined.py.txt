"""
final_combined.py
Combined inference pipeline: classification -> segmentation -> feature extraction -> therapy mapping

Usage:
 - Update MODEL paths and data paths below.
 - Run: python final_combined.py --image path/to/image.jpg
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# -----------------------
# Config - change these to your paths
# -----------------------
CLASSIFIER_MODEL = "inceptionv3_brain_classification.h5"
SEGMENTATION_MODEL = "unet_brain_segmentation.h5"
THERAPY_XLSX = "Therapy_Mapping.xlsx"  # Excel file mapping tumor features to therapies
IMG_SIZE_CLASS = (299, 299)
SEG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.55

# -----------------------
# load models
# -----------------------
def safe_load_models():
    # Dice functions for segmentation custom objects (if needed)
    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    def dice_loss(y_true, y_pred): return 1.0 - dice_coef(y_true,y_pred)

    classifier = load_model(CLASSIFIER_MODEL)
    segmentation = load_model(SEGMENTATION_MODEL, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    return classifier, segmentation

# -----------------------
# Classification helper
# -----------------------
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # adjust ordering to your training

def classify_image(classifier, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE_CLASS).astype('float32') / 255.0
    preds = classifier.predict(np.expand_dims(img_resized, 0))[0]
    idx = np.argmax(preds)
    confidence = preds[idx]
    label = CLASS_LABELS[idx] if confidence >= CONFIDENCE_THRESHOLD else "No Tumor"
    return label, confidence, preds

# -----------------------
# Segmentation helper
# -----------------------
def segment_image(seg_model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img_resized = cv2.resize(img, (SEG_SIZE, SEG_SIZE)).astype('float32') / 255.0
    inp = np.expand_dims(np.expand_dims(img_resized, -1), 0)
    pred = seg_model.predict(inp)[0,:,:,0]
    mask = (pred > 0.5).astype(np.uint8)
    # scale mask back to original image size
    mask_full = cv2.resize(mask.astype('uint8')*255, (w,h), interpolation=cv2.INTER_NEAREST)
    mask_full = (mask_full > 127).astype(np.uint8)
    return mask_full

# -----------------------
# Feature extraction
# -----------------------
def extract_features_from_mask(mask, gray_image):
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return {"Area": 0, "Shape": "None", "Texture": "None"}
    region = max(props, key=lambda r: r.area)
    area = region.area
    solidity = getattr(region, 'solidity', 0.0)
    eccentricity = getattr(region, 'eccentricity', 0.0)

    # simple shape classification
    if solidity > 0.90 and eccentricity < 0.4:
        shape = "Rounded"
    elif solidity < 0.85 and eccentricity > 0.7:
        shape = "Irregular"
    else:
        shape = "Lobulated"

    # texture using GLCM on the masked region
    # crop bounding box
    minr, minc, maxr, maxc = region.bbox
    crop = gray_image[minr:maxr, minc:maxc]
    crop = cv2.resize(crop, (64,64))
    # ensure uint8
    crop8 = (crop - crop.min()) / (crop.max() - crop.min() + 1e-8)
    crop8 = (crop8 * 255).astype('uint8')
    glcm = graycomatrix(crop8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    if contrast < 100 and homogeneity > 0.6:
        texture = "Homogeneous"
    elif contrast < 400:
        texture = "Heterogeneous"
    else:
        texture = "Necrotic"

    return {"Area": area, "Shape": shape, "Texture": texture, "Solidity": solidity, "Eccentricity": eccentricity}

# -----------------------
# Therapy mapping
# -----------------------
def load_therapy_mapping(xlsx_path):
    if not os.path.exists(xlsx_path):
        # Create a fallback mapping if file not present
        return [
            # mapping: (area_max, shape, texture, final_category, biomarkers, therapy, suitability)
            (1000, None, None, "Small (<1000)", "IDH1/2, MGMT, PD-L1", "Surgery, Temozolomide", "Moderate"),
            (3000, None, None, "Medium (1000-3000)", "TERT, EGFR, PD-L1", "Surgery, Radiation, EGFR inhibitors", "High"),
            (999999, None, None, "Large (>3000)", "PD-L1, NF2, CDKN2A/B", "Surgery, Clinical trials", "Moderate-High"),
        ]
    df = pd.read_excel(xlsx_path)
    # Expect columns: 'Area_Upper', 'Final Category', 'Key Biomarkers', 'Therapy Suggestions', 'Immunotherapy Suitability'
    mapping = []
    for _, row in df.iterrows():
        area_val = row.get('Tumor Feature (Pixel Area in pixel²)', None)
        mapping.append((area_val, row.get('Shape', None), row.get('Texture', None),
                        row.get('Final Category', None), row.get('Key Biomarkers', None),
                        row.get('Therapy Suggestions', None), row.get('Immunotherapy Suitability', None)))
    return mapping

def get_therapy_suggestion(mapping, features):
    area = features.get('Area', 0)
    # find first mapping entry with area >= area_threshold (mapping assumed sorted by area)
    for entry in mapping:
        area_threshold = entry[0]
        if area_threshold is None:
            continue
        if area <= area_threshold:
            return entry[3], entry[4], entry[5], entry[6]
    # fallback
    return "Unknown", "Unknown", "Observe", "Unknown"

# -----------------------
# Visualization helper
# -----------------------
def overlay_mask_on_image(img_path, mask):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colored = img_rgb.copy()
    colored[mask==1] = [255, 0, 0]  # red overlay
    blended = (0.6 * img_rgb + 0.4 * colored).astype('uint8')
    return blended

# -----------------------
# Run pipeline
# -----------------------
def run_single(image_path):
    print("Loading models...")
    classifier, seg = safe_load_models()
    print("Predicting classification...")
    tumor_type, conf, probs = classify_image(classifier, image_path)
    print(f"Predicted: {tumor_type} (confidence {conf:.2f})")

    print("Segmenting tumor...")
    mask = segment_image(seg, image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = extract_features_from_mask(mask, gray)
    print("Extracted features:", features)

    print("Loading therapy mapping...")
    mapping = load_therapy_mapping(THERAPY_XLSX)
    final_cat, biomarkers, therapy, suitability = get_therapy_suggestion(mapping, features)

    # Print results
    print("\n===== DDIS Result =====")
    print("Diagnosis:", tumor_type)
    print("Confidence:", f"{conf*100:.2f}%")
    print("Tumor Area (px²):", features.get('Area'))
    print("Shape:", features.get('Shape'))
    print("Texture:", features.get('Texture'))
    print("Final Category:", final_cat)
    print("Key Biomarkers:", biomarkers)
    print("Therapy Suggestions:", therapy)
    print("Immunotherapy Suitability:", suitability)

    # Visualization
    overlay = overlay_mask_on_image(image_path, mask)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(mask, cmap='gray'); plt.title("Mask"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis('off')
    plt.tight_layout()
    plt.savefig("ddis_output_visualization.png")
    plt.show()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    args = parser.parse_args()
    run_single(args.image)
