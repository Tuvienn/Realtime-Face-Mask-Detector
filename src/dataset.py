"""
Dataset module providing image generators and preprocessing functions.
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(
    train_dir: str,
    val_dir: str,
    target_size: tuple = (150, 150),
    batch_size: int = 32,
    augment: bool = True,
):
    """Tạo ImageDataGenerator cho train và validation set."""
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        target_size=target_size,
        class_mode='categorical',
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        batch_size=batch_size,
        target_size=target_size,
        class_mode='categorical',
    )

    return train_generator, val_generator


def preprocess_face(face_img: np.ndarray, target_size: tuple = (150, 150)) -> np.ndarray:
    """Preprocess single face crop image for inference batch."""
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_face, target_size)
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalized, axis=0)
    return batch
