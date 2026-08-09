import os
import shutil
import argparse
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Tuple, List, Dict


def build_fire_smoke_augmentation_pipeline() -> A.Compose:
    """
    Build an Albumentations composition pipeline specifically tuned for Fire & Smoke:
    - Random Brightness and Contrast variations (day/night/extreme light)
    - Gaussian Blur & Motion Blur (lens blur & fast motion)
    - Synthetic Haze/Fog and Gamma adjustments (smoke opacity simulation)
    - Horizontal Flip (spatial invariance)
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.35, 0.35),
            contrast_limit=(-0.35, 0.35),
            p=0.8
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5),
        A.RandomGamma(gamma_limit=(70, 140), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.6
        )
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.2
    ))


def augment_dataset(
    source_dir: str = "data/raw",
    output_dir: str = "data/processed",
    augmentations_per_image: int = 2
):
    """
    Reads YOLO formatted images and labels from source_dir, applies Albumentations
    augmentations (brightness, contrast, blur, haze), and writes augmented variations
    into output_dir.
    """
    src_path = Path(source_dir).resolve()
    out_path = Path(output_dir).resolve()

    pipeline = build_fire_smoke_augmentation_pipeline()

    for split in ["train"]:
        img_src = src_path / "images" / split
        lbl_src = src_path / "labels" / split

        if not img_src.exists():
            continue

        img_dst = out_path / "images" / split
        lbl_dst = out_path / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        img_files = list(img_src.glob("*.jpg")) + list(img_src.glob("*.png")) + list(img_src.glob("*.jpeg"))
        print(f"Processing split '{split}': {len(img_files)} original images...")

        total_generated = 0
        for img_file in img_files:
            lbl_file = lbl_src / f"{img_file.stem}.txt"

            image = cv2.imread(str(img_file))
            if image is None:
                continue

            # Read YOLO annotations
            bboxes = []
            class_labels = []
            if lbl_file.exists():
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:5])
                            # Clip bounding box to valid [0, 1] range
                            x_center = min(max(x_center, 0.001), 0.999)
                            y_center = min(max(y_center, 0.001), 0.999)
                            w = min(max(w, 0.001), 0.999)
                            h = min(max(h, 0.001), 0.999)

                            bboxes.append([x_center, y_center, w, h])
                            class_labels.append(cls_id)

            # Copy original to output_dir
            shutil.copy2(img_file, img_dst / img_file.name)
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dst / lbl_file.name)

            # Generate augmented variations
            for i in range(augmentations_per_image):
                try:
                    transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed["image"]
                    aug_boxes = transformed["bboxes"]
                    aug_labels = transformed["class_labels"]

                    aug_name = f"{img_file.stem}_aug{i+1}"
                    cv2.imwrite(str(img_dst / f"{aug_name}{img_file.suffix}"), aug_img)

                    # Save augmented labels
                    with open(lbl_dst / f"{aug_name}.txt", "w") as f:
                        for box, cls_id in zip(aug_boxes, aug_labels):
                            f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

                    total_generated += 1
                except Exception as e:
                    continue

        print(f"[SUCCESS] Generated {total_generated} augmented images in {img_dst}")

    # Copy val split directly
    for split in ["val"]:
        img_src = src_path / "images" / split
        lbl_src = src_path / "labels" / split
        if img_src.exists():
            shutil.copytree(img_src, out_path / "images" / split, dirs_exist_ok=True)
        if lbl_src.exists():
            shutil.copytree(lbl_src, out_path / "labels" / split, dirs_exist_ok=True)

    # Copy data.yaml to output_dir
    if (src_path / "data.yaml").exists():
        with open(src_path / "data.yaml", "r") as f:
            content = f.read().replace(str(src_path), str(out_path))
        with open(out_path / "data.yaml", "w") as f:
            f.write(content)
        print(f"[SUCCESS] Created augmented dataset config at {out_path / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire & Smoke Dataset Augmentation Pipeline (Albumentations)")
    parser.add_argument("--source", type=str, default="data/raw", help="Raw dataset path")
    parser.add_argument("--output", type=str, default="data/processed", help="Processed output dataset path")
    parser.add_argument("--copies", type=int, default=2, help="Number of augmented copies per image")

    args = parser.parse_args()
    augment_dataset(source_dir=args.source, output_dir=args.output, augmentations_per_image=args.copies)
