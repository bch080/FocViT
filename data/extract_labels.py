# coding=utf-8
"""
Extract OCT-MNIST labels from NPZ to CSV.

Outputs train_labels.csv, val_labels.csv, test_labels.csv
Each row: image_index, label
"""

import os
import numpy as np
import csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "..", "octmnist_224.npz")
OUTPUT_DIR = SCRIPT_DIR

SPLITS = {
    "train": ("train_images", "train_labels"),
    "val": ("val_images", "val_labels"),
    "test": ("test_images", "test_labels"),
}

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

CLASS_DESCRIPTIONS = {
    0: "CNV (Choroidal Neovascularization)",
    1: "DME (Diabetic Macular Edema)",
    2: "DRUSEN (Drusen)",
    3: "NORMAL (Normal)",
}


def extract_labels(npz_path: str, output_dir: str) -> None:
    """Extract labels from NPZ and save as CSV."""
    print(f"Loading NPZ from: {npz_path}")
    data = np.load(npz_path)

    for split_name, (images_key, labels_key) in SPLITS.items():
        labels = data[labels_key].squeeze()
        csv_path = os.path.join(output_dir, f"{split_name}_labels.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_index", "label"])
            for i, label in enumerate(labels):
                writer.writerow([i, int(label)])

        unique, counts = np.unique(labels, return_counts=True)
        dist = {CLASS_NAMES[u]: c for u, c in zip(unique, counts)}
        print(f"{split_name}: {len(labels)} samples | distribution: {dist}")
        print(f"  Saved to: {csv_path}")

    data.close()
    print(f"\nDone! Labels extracted to: {output_dir}")


def main():
    extract_labels(NPZ_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
