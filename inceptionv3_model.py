"""
inceptionv3_model.py
Training and evaluation script for tumor classification using InceptionV3.

Usage:
 - Prepare dataset directory with subfolders for each class:
   data/Training/Glioma/..., data/Training/Meningioma/..., data/Testing/...
 - Run: python inceptionv3_model.py --train
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# -----------------------
# Config / Paths (change these)
# -----------------------
TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"
MODEL_OUT = "inceptionv3_brain_classification.h5"
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 4
SEED = 42

# -----------------------
# Build model
# -----------------------
def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES):
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

# -----------------------
# Training routine
# -----------------------
def train_classifier(train_dir=TRAIN_DIR, test_dir=TEST_DIR, model_out=MODEL_OUT,
                     img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.15)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=img_size,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        subset='training',
                                                        seed=SEED)

    val_generator = train_datagen.flow_from_directory(train_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      subset='validation',
                                                      seed=SEED)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(train_generator.class_indices))
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(model_out, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]

    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=callbacks)

    # Evaluate on test set
    model = load_model(model_out)  # best model
    print("Evaluating on test set...")
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    labels = list(test_generator.class_indices.keys())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig('inception_confusion_matrix.png')

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.savefig('inception_training_plots.png')

    return model, history, test_generator

# -----------------------
# Inference helper
# -----------------------
def predict_image(model, img_path, size=IMG_SIZE):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    preds = model.predict(np.expand_dims(img,0))
    return preds[0]

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train classifier")
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--test_dir", type=str, default=TEST_DIR)
    parser.add_argument("--model_out", type=str, default=MODEL_OUT)
    args = parser.parse_args()

    if args.train:
        model, history, test_gen = train_classifier(args.train_dir, args.test_dir, args.model_out)
        print("Training complete. Model saved to:", args.model_out)
    else:
        print("No action specified. Use --train to train the classifier.")
