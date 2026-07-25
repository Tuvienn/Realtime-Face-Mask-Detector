"""
evaluate.py — Đánh giá model trên test set

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model ./best_model.keras --test-dir ./data/test --output ./outputs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ─────────────────────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────────────────────
IMAGE_SIZE  = (150, 150)
BATCH_SIZE  = 32
CLASS_NAMES = ["with_mask", "without_mask"]


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {save_path}")


def plot_training_history(history_path, save_dir):
    """Vẽ training curves nếu có history file."""
    import json
    if not os.path.exists(history_path):
        print(f"  ⚠️  No history file found at {history_path} — skipping training curves.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.get("loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.get("accuracy", []), label="Train Accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Training curves saved → {path}")


def evaluate(model_path: str, test_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\n🔍 Loading model from: {model_path}")
    model = load_model(model_path)

    # Load test data
    print(f"📂 Loading test data from: {test_dir}")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # QUAN TRỌNG: shuffle=False để giữ đúng thứ tự
    )

    class_names = list(test_generator.class_indices.keys())
    print(f"📌 Classes detected: {class_names}")
    print(f"📌 Total test samples: {test_generator.samples}\n")

    # Predict
    print("🤖 Running inference on test set...")
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    print("\n" + "=" * 55)
    print("📊 EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    report_path = os.path.join(output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("FACE MASK DETECTOR — EVALUATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test dir: {test_dir}\n")
        f.write(f"Test samples: {test_generator.samples}\n\n")
        f.write(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"\n  📄 Text report saved → {report_path}")

    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Training curves (nếu có)
    plot_training_history("./outputs/history.json", output_dir)

    print("\n✅ Evaluation complete!")
    print(f"   All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Mask Detector model")
    parser.add_argument("--model",    type=str, default="./best_model.keras", help="Path to trained model")
    parser.add_argument("--test-dir", type=str, default="./data/test",        help="Path to test dataset directory")
    parser.add_argument("--output",   type=str, default="./outputs",          help="Output directory for reports")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output,
    )
