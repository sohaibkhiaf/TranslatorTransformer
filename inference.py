#!/usr/bin/env python3
"""Run interactive or one-shot inference using a trained Transformer model.

This script is dedicated strictly for evaluating and testing models.
By default, it looks inside the 'models' directory for 'model_v1.pt',
but you can specify any model file name or path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from main import (
    ModelConfig,
    TransformerEncoderDecoder,
    Vocabulary,
    interactive_translation,
    load_checkpoint,
    translate,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "model_v1.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME


def resolve_model_path(model_arg: str | Path | None) -> Path:
    """Resolve model path, prioritizing the 'models' folder if a file name is given."""
    if not model_arg:
        target = DEFAULT_MODEL_PATH
    else:
        path = Path(model_arg)
        if path.is_file():
            return path.resolve()

        # Check inside 'models' folder first
        target = MODELS_DIR / path
        if not target.is_file():
            # Also check alternative folder named 'model' if present
            alt_target = BASE_DIR / "model" / path
            if alt_target.is_file():
                return alt_target.resolve()

    if not target.is_file():
        available_files = []
        for folder in (MODELS_DIR, BASE_DIR / "model"):
            if folder.is_dir():
                available_files.extend(str(f.name) for f in folder.glob("*.pt"))

        available_msg = (
            f"Available models in '{MODELS_DIR.name}/': {available_files}"
            if available_files
            else f"No .pt files found in '{MODELS_DIR}' directory."
        )
        raise FileNotFoundError(
            f"Model checkpoint not found at: {target}\n{available_msg}"
        )

    return target.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference tool for English-to-Italian Transformer model."
    )
    parser.add_argument(
        "model_file",
        nargs="?",
        default=DEFAULT_MODEL_NAME,
        help=f"Model file name in the 'models' directory or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="model_flag",
        default=None,
        help="Explicit model file name or path (overrides positional argument)",
    )
    parser.add_argument(
        "--sentence",
        "-s",
        type=str,
        default=None,
        help="Translate a single English sentence and exit instead of interactive mode",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Maximum decoding sequence length (defaults to model's max_len)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which model name/path to load
    model_input = args.model_flag or args.model_file
    model_path = resolve_model_path(model_input)

    # Select execution device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"Loading checkpoint from: {model_path} ...")
    model, source_vocab, target_vocab, checkpoint = load_checkpoint(model_path, device)

    epoch = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("current_validation_loss", None)
    if val_loss is not None:
        print(
            f"Loaded model successfully (Epoch {epoch} | "
            f"Validation loss: {val_loss:.4f} | Device: {device})"
        )
    else:
        print(f"Loaded model successfully (Epoch {epoch} | Device: {device})")

    # If single sentence provided, translate once and exit
    if args.sentence:
        print(f"\nEnglish: {args.sentence}")
        translation = translate(
            args.sentence,
            model,
            source_vocab,
            target_vocab,
            device,
            max_decode_len=args.max_len,
        )
        print(f"Italian: {translation}")
        return

    # Otherwise start interactive translation mode
    interactive_translation(model, source_vocab, target_vocab, device)


if __name__ == "__main__":
    main()
