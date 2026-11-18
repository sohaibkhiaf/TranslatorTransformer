#!/usr/bin/env python3
"""Export English and Italian vocabularies from checkpoints to text files (.txt) in the project root.

The sole source for vocabulary files is the 'checkpoints' folder.
If the .pkl files are not found inside 'checkpoints', no .txt files will be generated.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, List

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

ENGLISH_PKL = CHECKPOINTS_DIR / "english_vocabulary.pkl"
ITALIAN_PKL = CHECKPOINTS_DIR / "italian_vocabulary.pkl"


def extract_words(data: Any) -> List[str]:
    """Extract vocabulary tokens/words from saved format."""
    if isinstance(data, (list, tuple)):
        return [str(token) for token in data]
    if hasattr(data, "itos"):
        return [str(token) for token in data.itos]
    if isinstance(data, dict):
        if "itos" in data and isinstance(data["itos"], (list, tuple)):
            return [str(token) for token in data["itos"]]
        if all(isinstance(v, int) for v in data.values()):
            return [str(token) for token, _ in sorted(data.items(), key=lambda item: item[1])]
        if all(isinstance(k, int) for k in data.keys()):
            return [str(data[k]) for k in sorted(data.keys())]
        return [str(k) for k in data.keys()]
    raise TypeError(f"Unsupported vocabulary data structure: {type(data)}")


def export_pkl_to_txt(pkl_path: Path, output_dir: Path) -> Path:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    words = extract_words(data)
    output_file = output_dir / f"{pkl_path.stem}.txt"
    with output_file.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")

    print(f"Exported {len(words)} words from '{pkl_path.name}' -> {output_file.name}")
    return output_file


def export_vocabularies(
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    output_dir: Path = BASE_DIR,
) -> List[Path]:
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)

    target_files = [
        checkpoints_dir / "english_vocabulary.pkl",
        checkpoints_dir / "italian_vocabulary.pkl",
    ]

    existing_files = [f for f in target_files if f.is_file()]

    # If no .pkl files exist inside checkpoints, do not generate any .txt files
    if not existing_files:
        print(
            f"Error: No vocabulary .pkl files found in '{checkpoints_dir}'.\n"
            "No .txt files were created. Make sure training has run and saved vocabularies in checkpoints."
        )
        return []

    exported: List[Path] = []
    for pkl_file in existing_files:
        exported.append(export_pkl_to_txt(pkl_file, output_dir))

    # Notify if one of the languages is missing
    missing_files = [f for f in target_files if not f.is_file()]
    for mf in missing_files:
        print(f"Warning: '{mf.name}' was not found in '{checkpoints_dir}' and was not exported.")

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export English and Italian vocabularies from checkpoints to text files in the project root."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Path to checkpoints directory containing .pkl files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory to save exported text files (default: project root)",
    )

    args = parser.parse_args()
    results = export_vocabularies(args.checkpoints_dir, args.output_dir)
    if results:
        print(f"\nDone! Exported {len(results)} vocabulary file(s) to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
