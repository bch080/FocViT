# coding=utf-8
"""
Reduce OCT-MNIST dataset to a smaller subset with balanced class sampling.

Reduces:
- train: 5000 images (balanced across 4 classes)
- val:   1000 images (balanced across 4 classes)
- test:    10 images (balanced across 4 classes)

Outputs to octmnist_jpg_subset/ and *_labels_subset.csv with aligned data.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "..", "octmnist_224.npz")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "octmnist_jpg_subset")

TARGET_COUNTS = {
    "train": 5000,
    "val": 1000,
    "test": 10,
}

# per class
PER_CLASS = {k: v for k, v in TARGET_COUNTS.items()}

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

SPLITS = {
    "train": ("train_images", "train_labels"),
    "val": ("val_images", "val_labels"),
    "test": ("test_images", "test_labels"),
}


def balanced_sample_indices(labels: np.ndarray, per_class: int) -> list:
    """Return indices that balance classes, taking per_class from each."""
    num_classes = len(CLASS_NAMES)
    selected = []
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        chosen = class_indices[:per_class].tolist()
        selected.extend(chosen)
    return sorted(selected)


def subset_npz(npz_path: str, output_dir: str) -> None:
    """Create balanced subset of NPZ images and aligned CSV labels."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading NPZ from: {npz_path}")
    data = np.load(npz_path)

    csv_writers = {}
    csv_files = {}

    for split_name in SPLITS.keys():
        csv_path = os.path.join(output_dir, f"{split_name}_labels_subset.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["subset_index", "original_index", "label", "class_name"])
        csv_writers[split_name] = writer
        csv_files[split_name] = csv_file

    for split_name, (images_key, labels_key) in SPLITS.items():
        images = data[images_key]
        labels = data[labels_key].squeeze()

        split_dir = os.path.join(output_dir, split_name)
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

        indices = balanced_sample_indices(labels, PER_CLASS[split_name])
        total_selected = PER_CLASS[split_name] * len(CLASS_NAMES)
        print(f"\n{split_name}: selecting {total_selected} images ({PER_CLASS[split_name]} per class) from {len(labels)}")

        for local_idx, orig_idx in enumerate(tqdm(indices, desc=f"Saving {split_name}")):
            img = images[orig_idx]
            label = int(labels[orig_idx])
            class_name = CLASS_NAMES[label]

            img_pil = Image.fromarray(img, mode="L")
            filename = f"img_{local_idx:06d}.jpg"
            filepath = os.path.join(split_dir, class_name, filename)
            img_pil.save(filepath, quality=95)

            csv_writers[split_name].writerow([local_idx, orig_idx, label, class_name])

        class_counts = {c: len(os.listdir(os.path.join(split_dir, c))) for c in CLASS_NAMES}
        print(f"  {split_name} class distribution: {class_counts}")

    for f in csv_files.values():
        f.close()

    data.close()
    print(f"\nSubset created at: {output_dir}")
    print(f"You can now point your dataset loader to: {output_dir}")


def main():
    subset_npz(NPZ_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
