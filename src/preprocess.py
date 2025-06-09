from PIL import Image
from typing import List, Tuple


def load_and_preprocess_images(file_list: List[dict], size: Tuple[int, int] = (224, 224)) -> Tuple[List[Image.Image], List[str]]:
    """
    Load and resize images to a uniform size suitable for embedding.

    Args:
        file_list (List[dict]): List of metadata dicts from scan_images_metadata
        size (Tuple[int, int]): Target size for resizing (default is 224x224)

    Returns:
        Tuple[List[Image], List[str]]: List of PIL Images and corresponding file paths
    """
    images = []
    valid_paths = []

    for record in file_list:
        path = record["path"]
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(size)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    return images, valid_paths