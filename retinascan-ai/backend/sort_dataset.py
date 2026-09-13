"""
sort_dataset.py
---------------
Sorts Kaggle DR datasets (e.g. APTOS 2019) into class folders automatically,
supporting .png, .jpg, and .jpeg formats seamlessly.
"""

import argparse
import os
import shutil
import random
import csv

from model import CLASS_NAMES  # keep label order identical everywhere

ID_COL = "id_code"
LABEL_COL = "diagnosis"
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def find_image_file(images_dir: str, image_id: str, preferred_ext: str = "") -> str:
    """Helper function to search for the image file regardless of extension."""
    # Try preferred extension first if provided
    if preferred_ext:
        candidate = os.path.join(images_dir, f"{image_id}{preferred_ext}")
        if os.path.exists(candidate):
            return candidate

    # Fallback to checking all supported extensions
    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate

    return ""


def sort_dataset(csv_path: str, images_dir: str, output_dir: str, val_split: float, image_ext: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    random.seed(42)
    random.shuffle(rows)

    split_idx = int(len(rows) * (1 - val_split))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split_name, class_name), exist_ok=True)

        for row in split_rows:
            label_idx = int(row[LABEL_COL])
            class_name = CLASS_NAMES[label_idx]

            src = find_image_file(images_dir, row[ID_COL], image_ext)

            if src:
                filename = os.path.basename(src)
                dst = os.path.join(output_dir, split_name, class_name, filename)
                shutil.copy2(src, dst)
            else:
                print(f"WARNING: missing image for ID {row[ID_COL]} in {images_dir}, skipping")

    print(f"Done. {len(train_rows)} train images, {len(val_rows)} val images.")
    print(f"Output structure ready at: {output_dir}/train and {output_dir}/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Preferred extension e.g. .jpg or .png")
    args = parser.parse_args()

    sort_dataset(args.csv, args.images_dir, args.output_dir, args.val_split, args.image_ext)