"""
Model factory module supporting Custom CNN and MobileNetV2 Transfer Learning.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2


def build_custom_cnn(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Xây dựng Custom CNN 4-layer baseline model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax'),
    ], name="Custom_CNN_FaceMaskDetector")

    return model


def build_mobilenet_v2(
    input_shape: tuple = (150, 150, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    trainable_base: bool = False,
) -> tf.keras.Model:
    """Xây dựng Transfer Learning Model dựa trên MobileNetV2 (ImageNet weights)."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax'),
    ], name="MobileNetV2_FaceMaskDetector")

    return model


def get_model(architecture: str = "custom_cnn", **kwargs) -> tf.keras.Model:
    """Factory method để lấy model theo tên kiến trúc."""
    arch_lower = architecture.lower()
    if arch_lower in ["custom", "custom_cnn", "cnn"]:
        return build_custom_cnn(**kwargs)
    elif arch_lower in ["mobilenet", "mobilenetv2", "transfer_learning"]:
        return build_mobilenet_v2(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: '{architecture}'. Choose 'custom_cnn' or 'mobilenetv2'.")
