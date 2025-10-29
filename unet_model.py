"""
unet_model.py
U-Net based tumor segmentation training & inference script for DDIS project.

Usage:
 - Edit DATA_DIR and MASK_DIR to point to your images and masks (matched filenames).
 - Run training: python unet_model.py --train
 - For inference use functions at bottom or import this module.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Config / Paths (change these)
# -----------------------
DATA_DIR = "data/images"     # folder of input MRI images (png/jpg)
MASK_DIR = "data/masks"      # folder of binary masks with same filenames
MODEL_OUT = "unet_brain_segmentation.h5"
SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
SEED = 42

# -----------------------
# Helpers
# -----------------------
def load_images_and_masks(img_dir, mask_dir, size=SIZE):
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    images, masks = [], []
    for f_img in tqdm(image_files, desc="Loading images"):
        img_path = os.path.join(img_dir, f_img)
        mask_path = os.path.join(mask_dir, f_img)  # assume same filename
        if not os.path.exists(mask_path):
            # try mask with different suffix
            mask_candidates = [m for m in mask_files if os.path.splitext(m)[0] == os.path.splitext(f_img)[0]]
            if mask_candidates:
                mask_path = os.path.join(mask_dir, mask_candidates[0])
            else:
                continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.resize(img, (size, size))
        mask = cv2.resize(mask, (size, size))
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        images.append(np.expand_dims(img, -1))
        masks.append(np.expand_dims(mask, -1))
    X = np.array(images)
    y = np.array(masks)
    return X, y

# Dice metric & loss
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# -----------------------
# U-Net Model
# -----------------------
def build_unet(input_shape=(SIZE, SIZE, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# -----------------------
# Training routine
# -----------------------
def train_unet(X, y, out_model=MODEL_OUT, batch_size=BATCH_SIZE, epochs=EPOCHS):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED)
    model = build_unet(input_shape=(SIZE, SIZE, 1))
    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, 'accuracy'])
    model.summary()

    # Data augmentation
    data_gen_args = dict(rotation_range=15,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = SEED
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    callbacks = [
        ModelCheckpoint(out_model, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    steps_per_epoch = max(1, len(X_train) // batch_size)
    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)

    # Save history plots
    plt.figure()
    plt.plot(history.history.get('dice_coef', []), label='train_dice')
    plt.plot(history.history.get('val_dice_coef', []), label='val_dice')
    plt.legend(); plt.title('Dice Coefficient'); plt.savefig('unet_dice.png')

    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss'); plt.savefig('unet_loss.png')

    return model

# -----------------------
# Inference helper
# -----------------------
def predict_mask(model, image_path, size=SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    img_r = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    inp = np.expand_dims(np.expand_dims(img_r, -1), 0)
    pred = model.predict(inp)[0,:,:,0]
    mask = (pred > 0.5).astype(np.uint8)
    # Resize mask to original image size if needed
    return mask, pred

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train U-Net")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--mask_dir", type=str, default=MASK_DIR)
    parser.add_argument("--model_out", type=str, default=MODEL_OUT)
    args = parser.parse_args()

    if args.train:
        print("Loading dataset...")
        X, y = load_images_and_masks(args.data_dir, args.mask_dir)
        print("Shapes:", X.shape, y.shape)
        model = train_unet(X, y, out_model=args.model_out)
        print("Training finished. Model saved to:", args.model_out)
    else:
        print("No action specified. To train use --train")
