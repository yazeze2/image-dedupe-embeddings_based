from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from typing import List


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    """
    Load CLIP model and processor from Hugging Face.

    Args:
        model_name (str): model repo name
        device (str): "cpu" or "cuda"

    Returns:
        model, processor
    """
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def generate_clip_embeddings(images: List[Image.Image], model, processor, device: str = "cpu") -> torch.Tensor:
    """
    Generate normalized CLIP embeddings for a list of PIL images.

    Args:
        images (List[PIL.Image]): images to embed
        model: CLIP model
        processor: CLIP processor
        device: "cpu" or "cuda"

    Returns:
        torch.Tensor: normalized embeddings of shape (N, D)
    """
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()