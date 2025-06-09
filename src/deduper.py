# src/deduper.py

from pathlib import Path
import torch
from config import load_config, get_gallery_path
from scan import scan_images_metadata
from preprocess import load_and_preprocess_images
from embedder import load_clip_model, generate_clip_embeddings
from similarity import compute_similarity_matrix, group_duplicates
from file_ops import move_duplicate_images, save_log


def run_deduplication_for_months(year_month_pairs):
    config = load_config()
    base_dir = get_gallery_path(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip_model(device=device)

    for year, month in year_month_pairs:
        year_str = str(year)
        month_str = f"{month:02d}"
        print(f"\n🔍 Processing {year_str}/{month_str}...")
        month_dir = base_dir / year_str / month_str

        if not month_dir.exists():
            print(f"  -- [SKIPPED] Directory not found for {year_str}/{month_str}")
            save_log([], year_str, month_str, base_dir)
            continue

        metadata = scan_images_metadata(month_dir)
        if len(metadata) < 2:
            print(f"  -- [SKIPPED] Not enough images to deduplicate for {year_str}/{month_str}")
            save_log([], year_str, month_str, base_dir)
            continue

        images, valid_paths = load_and_preprocess_images(metadata)
        if len(images) < 2:
            print(f"  -- [SKIPPED] Could not load enough valid images for {year_str}/{month_str}")
            save_log([], year_str, month_str, base_dir)
            continue

        embeddings = generate_clip_embeddings(images, model, processor, device)
        sim_matrix = compute_similarity_matrix(embeddings)
        duplicate_groups = group_duplicates(sim_matrix, valid_paths, threshold=config.get("similarity_threshold", 0.95))

        if not duplicate_groups:
            print(f"  -- No duplicates found for {year_str}/{month_str}")
            save_log([], year_str, month_str, base_dir)
            continue

        log_entries = move_duplicate_images(duplicate_groups, month_dir)
        save_log(log_entries, year_str, month_str, base_dir)
        print(f"  -- Moved {len(log_entries)} duplicate files from {len(duplicate_groups)} group(s) in {year_str}/{month_str}.")
