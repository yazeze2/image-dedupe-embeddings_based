from pathlib import Path
from typing import List, Dict
import os

def scan_images_metadata(root_dir: Path) -> List[Dict]:
    """
    Scans all image files in the provided directory and returns metadata
    for each file that is locally available (size > 0).
    
    Args:
        root_dir (Path): Path to the folder to scan
    
    Returns:
        List[Dict]: List of metadata dictionaries (path, name, size, modified, etc.)
    """
    supported_exts = [".jpg", ".jpeg", ".png"]
    image_files = []

    for filepath in root_dir.rglob("*"):
        if filepath.suffix.lower() in supported_exts:
            try:
                stat = filepath.stat()
                if stat.st_size > 0:
                    image_files.append({
                        "path": str(filepath.resolve()),
                        "name": filepath.name,
                        "ext": filepath.suffix.lower(),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    return image_files