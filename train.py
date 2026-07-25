import os
import argparse
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from src.model import get_model
from src.dataset import get_data_generators

def main():
    parser = argparse.ArgumentParser(description="Train Face Mask Detector Model")
    parser.add_argument(
        "--arch",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "mobilenetv2"],
        help="Architecture to use: 'custom_cnn' or 'mobilenetv2'"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--train-dir", type=str, default="./data/train", help="Training dataset directory")
    parser.add_argument("--val-dir", type=str, default="./data/test", help="Validation dataset directory")
    parser.add_argument("--output", type=str, default="./best_model.keras", help="Path to save best model")
    args = parser.parse_args()

    IMAGE_SIZE = (150, 150)
    LOG_DIR = os.path.join(
        "logs", "fit",
        f"{args.arch}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    print(f"\n🚀 Initializing training with architecture: {args.arch}")
    print(f"📂 Train dir: {args.train_dir} | Val dir: {args.val_dir}")
    print(f"⚙️ Epochs: {args.epochs} | Batch size: {args.batch_size}\n")

    # 1. Data Generators
    train_generator, val_generator = get_data_generators(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        augment=True
    )

    # 2. Build Model
    model = get_model(architecture=args.arch, input_shape=(*IMAGE_SIZE, 3), num_classes=2)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        args.output,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    # 4. Fit Model
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, tensorboard],
    )
    print(f"\n✅ Training complete! Best model saved to: {args.output}")

if __name__ == "__main__":
    main()


