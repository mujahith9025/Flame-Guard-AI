import os
import sys
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from src.inference.detector import FireSmokeDetector

# Force UTF-8 standard output encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def draw_ground_truth(image: np.ndarray, label_path: Path, class_names: dict):
    """Draw ground truth bounding boxes in green."""
    if not label_path.exists():
        return image

    h, w, _ = image.shape
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:5])

            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            label_text = f"GT: {class_names.get(cls_id, cls_id)}"
            color = (0, 255, 0)  # Green for Ground Truth

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return image


def generate_gt_vs_pred_comparison(
    model_path: str = "best.pt",
    val_img_dir: str = "data/raw/images/val",
    val_lbl_dir: str = "data/raw/labels/val",
    output_grid_path: str = "data/eval_predictions_vs_gt.png",
    num_samples: int = 10
):
    """
    Generate side-by-side comparison (Ground Truth vs Prediction) for 10 validation images.
    """
    img_dir = Path(val_img_dir)
    lbl_dir = Path(val_lbl_dir)
    
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    img_files = [f for f in img_files if f.name != ".gitkeep"]

    if not img_files:
        print(f"No validation images found in {img_dir}")
        return

    detector = FireSmokeDetector(model_path=model_path, conf_threshold=0.25)
    class_names = detector.model.names

    selected_files = img_files[:min(num_samples, len(img_files))]

    fig, axes = plt.subplots(len(selected_files), 2, figsize=(12, 3.5 * len(selected_files)))
    if len(selected_files) == 1:
        axes = np.array([axes])

    for i, img_path in enumerate(selected_files):
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        # Ground Truth
        gt_img = img_orig.copy()
        gt_img = draw_ground_truth(gt_img, lbl_path, class_names)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Prediction
        pred_img, _ = detector.process_frame(img_orig.copy())
        pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(gt_rgb)
        axes[i, 0].set_title(f"Sample {i+1}: Ground Truth ({img_path.name[:20]})", fontsize=10, color="green", fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title(f"Sample {i+1}: Model Prediction", fontsize=10, color="blue", fontweight="bold")
        axes[i, 1].axis("off")

    plt.suptitle("Validation Set Evaluation: Ground Truth (Left) vs YOLOv8 Prediction (Right)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_grid_path), exist_ok=True)
    plt.savefig(output_grid_path, dpi=200, bbox_inches="tight")
    print(f"[SUCCESS] Saved 10-sample GT vs Prediction evaluation grid to: {output_grid_path}")
    plt.close()


def copy_eval_plots(runs_dir: str = "runs/detect/runs/train/fire_smoke_yolov8s"):
    """Copy confusion matrix and PR curves into data/ directory."""
    src_dir = Path(runs_dir)
    target_dir = Path("data")

    for plot_name in ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"]:
        src_file = src_dir / plot_name
        if src_file.exists():
            shutil.copy2(src_file, target_dir / plot_name)
            print(f"[SUCCESS] Copied {plot_name} to {target_dir / plot_name}")


def main():
    print("\n==========================================")
    print("      EVALUATING MODEL ON VALIDATION SET  ")
    print("==========================================")
    
    # 1. Copy plots
    copy_eval_plots()

    # 2. Generate side-by-side comparison for 10 validation images
    generate_gt_vs_pred_comparison(
        model_path="best.pt",
        val_img_dir="data/raw/images/val",
        val_lbl_dir="data/raw/labels/val",
        output_grid_path="data/eval_predictions_vs_gt.png",
        num_samples=10
    )


if __name__ == "__main__":
    main()
