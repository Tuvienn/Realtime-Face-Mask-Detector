"""
split_dataset.py — Tách dataset thành train / val / test

Usage:
    python scripts/split_dataset.py --source ./train --output ./data
    python scripts/split_dataset.py --source ./train --output ./data --train 0.7 --val 0.15 --test 0.15

Output structure:
    data/
    ├── train/
    │   ├── with_mask/
    │   └── without_mask/
    ├── val/
    │   ├── with_mask/
    │   └── without_mask/
    └── test/
        ├── with_mask/
        └── without_mask/
"""

import os
import shutil
import random
import argparse


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    random.seed(seed)

    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]
    if not classes:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    print(f"\n📂 Source: {source_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Split: train={train_ratio:.0%} | val={val_ratio:.0%} | test={test_ratio:.0%}")
    print(f"🌱 Seed: {seed}\n")

    total_moved = {"train": 0, "val": 0, "test": 0}

    for class_name in classes:
        class_source = os.path.join(source_dir, class_name)
        files = [
            f for f in os.listdir(class_source)
            if os.path.isfile(os.path.join(class_source, f))
        ]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)
        n_test  = n_total - n_train - n_val  # remainder goes to test

        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }

        print(f"  Class: {class_name} — total: {n_total}")
        for split_name, split_files in splits.items():
            dest_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for fname in split_files:
                src  = os.path.join(class_source, fname)
                dest = os.path.join(dest_dir, fname)
                shutil.copy2(src, dest)

            total_moved[split_name] += len(split_files)
            print(f"    → {split_name}: {len(split_files)} files → {dest_dir}")

    print("\n✅ Split complete!")
    print(f"   Train: {total_moved['train']} images")
    print(f"   Val:   {total_moved['val']} images")
    print(f"   Test:  {total_moved['test']} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--source",  type=str, default="./train",  help="Source directory with class subdirs")
    parser.add_argument("--output",  type=str, default="./data",   help="Output directory for splits")
    parser.add_argument("--train",   type=float, default=0.70,     help="Train ratio (default: 0.70)")
    parser.add_argument("--val",     type=float, default=0.15,     help="Val ratio (default: 0.15)")
    parser.add_argument("--test",    type=float, default=0.15,     help="Test ratio (default: 0.15)")
    parser.add_argument("--seed",    type=int,   default=42,        help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
