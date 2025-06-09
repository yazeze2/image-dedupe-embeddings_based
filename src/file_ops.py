# src/file_ops.py

import shutil
from pathlib import Path
from typing import List, Dict
import pandas as pd


def move_duplicate_images(duplicate_groups: List[List[str]], month_dir: Path) -> List[Dict]:
    """
    Move duplicate files (excluding first image in each group) to a `duplicates/` folder.

    Args:
        duplicate_groups (List[List[str]]): List of duplicate groups
        month_dir (Path): Path to the current month folder

    Returns:
        List[Dict]: Log entries of moved files
    """
    duplicates_dir = month_dir / "duplicates"
    duplicates_dir.mkdir(exist_ok=True)

    log_entries = []

    for group in duplicate_groups:
        keep = group[0]  # keep the first image
        to_move = group[1:]  # move the rest

        for src_path_str in to_move:
            src = Path(src_path_str)
            dst = duplicates_dir / src.name
            i = 1
            while dst.exists():
                dst = duplicates_dir / f"{src.stem}_{i}{src.suffix}"
                i += 1

            try:
                shutil.move(str(src), str(dst))
                log_entries.append({
                    "year": month_dir.parent.name,
                    "month": month_dir.name,
                    "group_leader": keep,
                    "moved_file": str(src),
                    "new_location": str(dst)
                })
            except Exception as e:
                print(f"Failed to move {src}: {e}")

    return log_entries


def save_log(log_entries: List[Dict], year: str, month: str, gallery_path: Path, logs_root: Path = Path("../logs")) -> None:
    """
    Save deduplication log entries to a CSV file under logs/{year}/dedupe_log.csv
    and also in OneDrive/year/duplicates_summary/

    Args:
        log_entries (List[Dict]): Entries to log
        year (str): Year of the log entries
        month (str): Month of the log entries
        gallery_path (Path): Base path to OneDrive gallery
        logs_root (Path): Root directory to save logs into
    """
    year_log_dir = logs_root.resolve()
    # year_log_dir.mkdir(parents=True, exist_ok=True)
    fname = "dedupe_log" + "_" + str(year) + ".csv"
    log_file = year_log_dir / fname

    # Path to save in OneDrive/year/duplicates_summary
    alt_log_file = gallery_path / year / "duplicates_summary" / "dedupe_log.csv"
    alt_log_file.parent.mkdir(parents=True, exist_ok=True)

    if log_entries:
        df = pd.DataFrame(log_entries)
        if log_file.exists():
            df_existing = pd.read_csv(log_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(log_file, index=False)
        df.to_csv(alt_log_file, index=False)
        print(f"  -- Summary updated for {year}/{month}")
    else:
        note = pd.DataFrame([{
            "year": year,
            "month": month,
            "group_leader": "None",
            "moved_file": "None",
            "new_location": "No duplicates found"
        }])
        if log_file.exists():
            df_existing = pd.read_csv(log_file)
            note = pd.concat([df_existing, note], ignore_index=True)
        note.to_csv(log_file, index=False)
        note.to_csv(alt_log_file, index=False)
        print(f"  -- Summary updated for {year}/{month} (no duplicates found)")
