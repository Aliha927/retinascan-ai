"""
sort_dataset.py
---------------
Most Kaggle DR datasets (e.g. APTOS 2019) ship as:
    train_images/  (all images in one folder)
    train.csv      (columns: id_code, diagnosis [0-4])

train.py needs a folder-per-class structure instead. This script converts
one into the other, with an automatic train/val split.

USAGE:
    python sort_dataset.py \
        --csv train.csv \
        --images_dir train_images \
        --output_dir dataset \
        --val_split 0.15

CSV format expected: two columns - image filename (without extension) and
an integer label 0-4, where:
    0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
(This is the standard APTOS 2019 label scheme. If your dataset uses a
different column layout, adjust ID_COL / LABEL_COL below.)
"""

import argparse
import os
import shutil
import random
import csv

from model import CLASS_NAMES  # keep label order identical everywhere

ID_COL = "id_code"
LABEL_COL = "diagnosis"


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
            src = os.path.join(images_dir, f"{row[ID_COL]}{image_ext}")
            dst = os.path.join(output_dir, split_name, class_name, f"{row[ID_COL]}{image_ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"WARNING: missing image {src}, skipping")

    print(f"Done. {len(train_rows)} train images, {len(val_rows)} val images.")
    print(f"Output structure ready at: {output_dir}/train and {output_dir}/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--image_ext", type=str, default=".png", help="e.g. .png or .jpg")
    args = parser.parse_args()

    sort_dataset(args.csv, args.images_dir, args.output_dir, args.val_split, args.image_ext)
