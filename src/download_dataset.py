import os
import sys
import json
import shutil
import zipfile
import argparse
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RoboflowDatasetDownloader")


def download_with_roboflow_sdk(api_key: str, workspace: str, project: str, version: int) -> Path:
    """Download dataset using official Roboflow SDK."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    downloaded_location = rf_project.version(version).download("yolov8")
    return Path(downloaded_location.location)


def download_with_direct_api(api_key: str, workspace: str, project: str, version: int, tmp_dir: Path) -> Path:
    """Download dataset directly via Roboflow REST API as zip."""
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={api_key}"
    logger.info(f"Querying Roboflow API: https://api.roboflow.com/{workspace}/{project}/{version}/yolov8")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                data = json.loads(resp.read().decode("utf-8"))
                download_link = data.get("export", {}).get("link") or data.get("gdrive")
                if not download_link:
                    raise ValueError(f"Unexpected API response: {data}")
            else:
                # Direct file stream
                download_link = resp.geturl()
    except Exception as e:
        logger.error(f"Failed to query Roboflow API link: {e}")
        # Try direct link format
        download_link = f"https://universe.roboflow.com/ds/{project}?key={api_key}"

    zip_path = tmp_dir / "dataset.zip"
    logger.info(f"Downloading ZIP package from Roboflow...")
    urllib.request.urlretrieve(download_link, zip_path)

    extract_path = tmp_dir / "extracted"
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path


def organize_yolo_dataset(source_path: Path, raw_dir: Path):
    """Organize extracted files into data/raw/ standard YOLO structure."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Organizing dataset from {source_path} into {raw_dir}...")

    # Look for data.yaml
    yaml_files = list(source_path.rglob("data.yaml"))
    if yaml_files:
        base_dir = yaml_files[0].parent
    else:
        base_dir = source_path

    # Copy splits
    for split in ["train", "valid", "val", "test"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue

        target_split = "val" if split in ["valid", "val"] else split

        # Copy images
        img_src = split_dir / "images"
        if not img_src.exists() and (split_dir).is_dir():
            img_src = split_dir

        if img_src.exists():
            img_dst = raw_dir / "images" / target_split
            img_dst.mkdir(parents=True, exist_ok=True)
            for f in img_src.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    shutil.copy2(f, img_dst / f.name)

        # Copy labels
        lbl_src = split_dir / "labels"
        if lbl_src.exists():
            lbl_dst = raw_dir / "labels" / target_split
            lbl_dst.mkdir(parents=True, exist_ok=True)
            for f in lbl_src.glob("*.txt"):
                shutil.copy2(f, lbl_dst / f.name)

    # Copy data.yaml
    data_yaml_src = base_dir / "data.yaml"
    if data_yaml_src.exists():
        shutil.copy2(data_yaml_src, raw_dir / "data.yaml")
        logger.info(f"Copied data.yaml to {raw_dir / 'data.yaml'}")


def download_roboflow_dataset(
    api_key: str,
    workspace: str = "senior-design-project-cavtw",
    project: str = "fire-and-smoke-detection-yolov8",
    version: int = 1,
    target_dir: str = "data/raw"
):
    raw_dir = Path(target_dir).resolve()
    tmp_dir = raw_dir.parent / "_tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = None

    # Method 1: Roboflow SDK
    try:
        logger.info("Attempting download via Roboflow SDK...")
        extracted_path = download_with_roboflow_sdk(api_key, workspace, project, version)
    except Exception as e:
        logger.warning(f"Roboflow SDK download failed or unavailable ({e}). Falling back to Direct REST API...")

    # Method 2: Direct REST API fallback
    if not extracted_path:
        try:
            extracted_path = download_with_direct_api(api_key, workspace, project, version, tmp_dir)
        except Exception as e:
            logger.error(f"Direct REST API download failed: {e}")
            raise e

    if extracted_path:
        organize_yolo_dataset(extracted_path, raw_dir)

    # Cleanup temp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("🎉 Dataset download and setup completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fire & Smoke YOLOv8 Dataset from Roboflow Universe")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""), help="Roboflow API Key")
    parser.add_argument("--workspace", type=str, default="senior-design-project-cavtw", help="Roboflow Workspace")
    parser.add_argument("--project", type=str, default="fire-and-smoke-detection-yolov8", help="Roboflow Project")
    parser.add_argument("--version", type=int, default=1, help="Roboflow Dataset Version")
    parser.add_argument("--target", type=str, default="data/raw", help="Target dataset directory")

    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key provided. Usage: python -m src.download_dataset --api-key YOUR_KEY")
        sys.exit(1)

    download_roboflow_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        target_dir=args.target
    )
