# Photo Deduplication with Open AI's CLIP Embeddings

This project implements a scalable image deduplication pipeline designed for large, chronologically organized photo libraries (a 1TB+ gallery stored on OneDrive). It uses CLIP-based image embeddings and cosine similarity to identify redundant images and isolate a single representative per group. The system is modular, auditable, and intended to be extended into a full photo management application.

**Open AI's CLIP** was selected for its ability to capture **semantic-level similarity** between images, rather than relying on pixel-level or hash-based matching. Its ViT-based encoder tokenizes and embeds images into a high-dimensional space where similar scenes, people, or objects cluster naturally, even with slight variations in lighting, or background.This makes it particularly effective for detecting duplicates in real-world photo libraries, where exact copies are rare, but near-duplicates are frequent.

## Motivation

Between burst shots, toddler motion blur, and accidental long-presses, the average modern photo library includes not just moments — but hundreds of near-identical moments. This project was implemented out of a very real need: efficiently de-cluttering a massive family photo archive without spending hours manually reviewing tiny variations in smile angles.
The goal is to automate what’s become an absurd digital ritual: scrolling through 47 nearly identical pictures from that one backyard barbecue and thinking, "I’ll clean this up someday." Spoiler: you won’t. That’s what this pipeline is for.

## Key Features

Deduplication operates at the year/month folder level

 - Embedding generation using openai/clip-vit-base-patch32. Images are tokenized using a vision transformer (ViT) encoder within CLIP, which converts visual inputs into a sequence of patch-level embeddings. This tokenization enables semantic-level similarity detection, beyond pixel-level differences.
    - In CLIP, the image is first resized (typically 224×224).
    - It’s then split into smaller patches.
    - Each patch is flattened and projected into an embedding space, similar to how text tokens are embedded.
    - These patch embeddings form a sequence, which is passed to the vision transformer (ViT). In essence, this lets the model treat an image as a sequence just like it does with text.

 - Duplicate detection using cosine similarity + Union-Find grouping

 - Retains one image per group and relocates others to duplicates/

 - Produces structured logs both locally and within the OneDrive gallery

 - Designed for integration with a notebook or future web interface

## Directory Structure

```
image-dedupe-embeddings-based/
├── config.yaml
├── notebooks/
│   └── run_deduplication.ipynb
│   └── exploration.ipynb
│   └── explore_end_to_end.ipynb
├── logs/
├── src/
│   ├── config.py
│   ├── scan.py
│   ├── preprocess.py
│   ├── embedder.py
│   ├── similarity.py
│   ├── file_ops.py
│   └── deduper.py
├── requirements.txt
└── README.md
```

## How It Works

 - Specify a list of (year, month) pairs.

 - Images are preprocessed and embedded using CLIP.

 - Cosine similarity identifies semantically similar pairs.

 - Union-Find groups images above a configurable similarity threshold.

 - Only one image per group is retained; others are moved to a duplicates/ folder.

 - Results are logged to both logs/ and OneDrive/Pictures/{year}/duplicates_summary/.

### Example Usage

```
python from deduper import run_deduplication_for_months run_deduplication_for_months([(2024, 6), (2024, 7)])
```
```
Output:
            Moved 18 duplicate files from 9 group(s) in 2024/06.
            Summary updated for 2024/06
```

### Requirements

 - Python 3.8+
 - Folder hierarchy structured as Pictures/{year}/{month}/
 - Jupyter (for exploratory runs and logic transparency)
 - Optional GPU acceleration for embedding generation

## Future Enhancements

- Web-based UI (e.g., Streamlit or Gradio) for visual inspection and manual override

- Hybrid embedding strategies combining CLIP and perceptual hashing

- CLI interface for batch job execution

- Integration with unit testing frameworks and CI pipeline





