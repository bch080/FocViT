# coding=utf-8
"""
OCT-MNIST Dataset NPZ to JPG Converter

Converts OCT-MNIST grayscale images from NPZ format to organized JPG folders.
Each split (train/val/test) is saved in its own folder with subfolders per class.

NPZ Structure:
- train_images: (97477, 224, 224), uint8
- train_labels: (97477, 1), uint8 (classes 0-3)
- val_images: (10832, 224, 224), uint8
- val_labels: (10832, 1), uint8
- test_images: (1000, 224, 224), uint8
- test_labels: (1000, 1), uint8
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "..", "octmnist_224.npz")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "octmnist_jpg")

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

SPLITS = {
    "train": ("train_images", "train_labels"),
    "val": ("val_images", "val_labels"),
    "test": ("test_images", "test_labels"),
}


def npz_to_jpg(npz_path: str, output_dir: str) -> None:
    """Convert NPZ dataset to JPG folders."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading NPZ from: {npz_path}")
    data = np.load(npz_path)

    for split_name, (images_key, labels_key) in SPLITS.items():
        images = data[images_key]
        labels = data[labels_key].squeeze()

        split_dir = os.path.join(output_dir, split_name)
        print(f"\nProcessing {split_name}: {len(images)} images")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        for i, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc=f"Saving {split_name}")):
            img_pil = Image.fromarray(img, mode="L")
            class_name = CLASS_NAMES[label]
            filename = f"img_{i:06d}.jpg"
            filepath = os.path.join(split_dir, class_name, filename)
            img_pil.save(filepath, quality=95)

        class_counts = {c: len(os.listdir(os.path.join(split_dir, c))) for c in CLASS_NAMES}
        print(f"  {split_name} class distribution: {class_counts}")

    data.close()
    print(f"\nConversion complete! Output: {output_dir}")


def main():
    npz_to_jpg(NPZ_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
