import os
import sys
import random
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Set standard output encoding to UTF-8 for Windows compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def load_dataset_config(yaml_path="data/raw/data.yaml"):
    """Load class names from data.yaml."""
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            names = cfg.get("names", {0: "Fire", 1: "Person", 2: "Smoke"})
            if isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
            return names
    return {0: "Fire", 1: "Person", 2: "Smoke"}


def compute_class_distribution(labels_dir: Path, class_names: dict):
    """Iterate through all label files and count instances per class."""
    counts = Counter()
    label_files = list(labels_dir.glob("*.txt"))

    for lbl_file in label_files:
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1

    print("\n==========================================")
    print("DATASET CLASS DISTRIBUTION (TRAIN SET)")
    print("==========================================")
    total_instances = sum(counts.values())
    for cls_id, name in sorted(class_names.items()):
        cnt = counts.get(cls_id, 0)
        percentage = (cnt / total_instances * 100) if total_instances > 0 else 0
        print(f"  • Class {cls_id} [{name}]: {cnt} instances ({percentage:.1f}%)")
    print(f"Total Annotated Bounding Boxes: {total_instances}")
    print("==========================================\n")
    return counts


def draw_yolo_boxes(image: np.ndarray, label_path: Path, class_names: dict):
    """Draw bounding boxes from YOLO format text file on the image."""
    if not label_path.exists():
        return image

    h, w, _ = image.shape
    colors = {
        0: (255, 50, 50),   # Red for Fire
        1: (50, 150, 255),  # Blue for Person
        2: (128, 128, 128)  # Gray for Smoke
    }

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:5])

            # Convert normalized YOLO coordinates to pixel values
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            color = colors.get(cls_id, (0, 255, 0))
            label_text = class_names.get(cls_id, f"Class {cls_id}")

            # Draw rectangle and text background
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                image, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

    return image


def visualize_random_samples(
    images_dir: Path,
    labels_dir: Path,
    class_names: dict,
    num_samples: int = 9,
    output_path: str = "data/dataset_preview.png"
):
    """Select random images and render 3x3 visualization grid."""
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    image_files = [f for f in image_files if f.name != ".gitkeep"]

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, img_path in enumerate(selected_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        annotated_img = draw_yolo_boxes(img_rgb, lbl_path, class_names)

        axes[idx].imshow(annotated_img)
        axes[idx].set_title(img_path.name[:25], fontsize=9)
        axes[idx].axis("off")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Fire & Smoke Dataset - 9 Random Training Samples with Annotations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Dataset visualization saved to: {output_path}")
    plt.close()


def main():
    dataset_raw = Path("data/raw")
    images_train = dataset_raw / "images" / "train"
    labels_train = dataset_raw / "labels" / "train"
    data_yaml = dataset_raw / "data.yaml"

    class_names = load_dataset_config(data_yaml)

    # 1. Compute and print class distribution
    compute_class_distribution(labels_train, class_names)

    # 2. Visualize 9 random samples
    visualize_random_samples(images_train, labels_train, class_names, num_samples=9)


if __name__ == "__main__":
    main()
