import os
import sys
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# Set UTF-8 output encoding for Windows compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelExporter")


def export_yolo_model(
    weights_path: str = "best.pt",
    formats: list = ["onnx", "openvino"],
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False
):
    """
    Export trained YOLOv8 model to high-performance edge deployment formats:
    - ONNX (.onnx) for universal cross-platform CPU/GPU runtimes.
    - OpenVINO (_openvino_model) for Intel CPU/iGPU acceleration.
    - TensorRT (.engine) for NVIDIA GPU / Jetson acceleration (requires CUDA).

    Args:
        weights_path (str): Path to PyTorch model checkpoint (.pt).
        formats (list): Formats to export ('onnx', 'openvino', 'engine').
        imgsz (int): Image dimension.
        half (bool): Use FP16 half precision.
        dynamic (bool): Enable dynamic shape dimensions.
    """
    weights_file = Path(weights_path)
    if not weights_file.exists():
        logger.error(f"Weights file '{weights_path}' not found.")
        return False

    logger.info(f"Loading PyTorch checkpoint from '{weights_file}'...")
    model = YOLO(str(weights_file))

    exported_paths = {}

    for fmt in formats:
        logger.info(f"--------------------------------------------------")
        logger.info(f"🚀 Exporting model to format: '{fmt.upper()}' (imgsz={imgsz}, half={half})...")
        try:
            exported_file = model.export(
                format=fmt,
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=(fmt == "onnx")
            )
            exported_paths[fmt] = str(exported_file)
            logger.info(f"✅ Export successful for format '{fmt}': {exported_file}")
        except Exception as e:
            logger.error(f"❌ Failed to export to format '{fmt}': {e}")

    print("\n==========================================")
    print("      MODEL EXPORT SUMMARY REPORT         ")
    print("==========================================")
    for fmt, path in exported_paths.items():
        print(f"  • {fmt.upper()}: {path}")
    print("==========================================\n")

    return exported_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Trained YOLOv8 Model to ONNX, OpenVINO, TensorRT")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to PyTorch model weights")
    parser.add_argument("--format", type=str, nargs="+", default=["onnx", "openvino"], help="Export formats: onnx, openvino, engine")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--half", action="store_true", help="Export with FP16 half precision")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic image input shapes")

    args = parser.parse_args()

    export_yolo_model(
        weights_path=args.weights,
        formats=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic
    )
