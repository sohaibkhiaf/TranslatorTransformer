#!/usr/bin/env python3
"""Train a small English-to-Italian Transformer (word-level tokenization).

The implementation defines multi-head attention explicitly so that the
encoder self-attention, decoder causal self-attention, and encoder-decoder
cross-attention masks are visible and easy to study.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "translation_dataset.csv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Reproducibility and tokenization
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    """Unicode-aware word tokenization; punctuation is kept as separate tokens."""
    return TOKEN_PATTERN.findall(text.strip().lower())


def detokenize(tokens: Sequence[str]) -> str:
    """Join word tokens and attach common punctuation naturally."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s+([)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was previously decoded as Windows-1252."""
    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        # Text that was already decoded correctly cannot safely be repaired.
        return text


class Vocabulary:
    def __init__(self, tokens: Sequence[str]):
        self.itos = list(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        for token in SPECIAL_TOKENS:
            if token not in self.stoi:
                raise ValueError(f"Vocabulary is missing reserved token: {token}")

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1) -> "Vocabulary":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        # Sorting makes vocabulary creation deterministic across runs.
        words = sorted(
            token for token, count in counts.items()
            if count >= min_frequency and token not in SPECIAL_TOKENS
        )
        return cls(SPECIAL_TOKENS + words)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokenize(text)]

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]


# ---------------------------------------------------------------------------
# Dataset and batching
# ---------------------------------------------------------------------------

def read_parallel_corpus(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place translation_dataset.csv in the base directory or pass --data PATH."
        )

    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        # The bundled corpus is CSV. Sniff the delimiter so compatible TSV or
        # semicolon-separated exports can still be supplied with --data.
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error as error:
            raise ValueError("Could not determine the dataset delimiter.") from error

        reader = csv.DictReader(file, dialect=dialect)
        required = {"english", "italian"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Dataset must contain the headers: english,italian."
            )
        for row_number, row in enumerate(reader, start=2):
            source = repair_mojibake((row.get("english") or "").strip())
            target = repair_mojibake((row.get("italian") or "").strip())
            if not source or not target:
                raise ValueError(f"Empty sentence found at dataset row {row_number}.")
            pairs.append((source, target))

    if len(pairs) < 10:
        raise ValueError(
            f"Only {len(pairs)} sentence pairs were found. Use at least 10 pairs; "
            "a larger corpus is strongly recommended."
        )
    return pairs


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        max_len: int,
    ):
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for source_text, target_text in pairs:
            source_ids = source_vocab.encode(source_text) + [source_vocab.eos_id]
            target_ids = [target_vocab.bos_id] + target_vocab.encode(target_text) + [target_vocab.eos_id]
            if len(source_ids) > max_len or len(target_ids) > max_len:
                continue
            self.examples.append((torch.tensor(source_ids), torch.tensor(target_ids)))
        if not self.examples:
            raise ValueError(
                f"No examples fit within --max-len={max_len}. Increase --max-len."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def make_collate_fn(pad_id: int):
    def collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        sources, targets = zip(*batch)
        source_batch = pad_sequence(sources, batch_first=True, padding_value=pad_id)
        target_batch = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        return source_batch, target_batch
    return collate


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Boolean masks use True to mean "do not attend".
    key_padding_mask: [batch, key_length]
    attention_mask:   [query_length, key_length] (e.g. causal upper triangle)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [B, heads, length, head_dim]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Scaled dot-product attention: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(
            query.size(0), query.size(1), self.d_model
        )
        return self.out_proj(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings added to token embeddings."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = token_embeddings.shape
        if sequence_length > self.embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_len="
                f"{self.embedding.num_embeddings}. Increase --max-len."
            )
        positions = torch.arange(sequence_length, device=token_embeddings.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        return token_embeddings + self.embedding(positions)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        # Pre-LayerNorm is generally stable for deeper Transformer stacks.
        self.norm_attention = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ff_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, source_padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(x)
        x = x + self.residual_dropout(
            self.self_attention(
                normalized, normalized, normalized,
                key_padding_mask=source_padding_mask,
            )
        )
        x = x + self.residual_dropout(self.ffn(self.norm_ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm_self_attention = nn.LayerNorm(d_model)
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_cross_attention = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ff_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        target_padding_mask: torch.Tensor,
        source_padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1) Masked self-attention: each target position can see itself and
        #    earlier target positions, but never future target positions.
        normalized = self.norm_self_attention(x)
        x = x + self.residual_dropout(
            self.masked_self_attention(
                normalized, normalized, normalized,
                key_padding_mask=target_padding_mask,
                attention_mask=causal_mask,
            )
        )

        # 2) Cross-attention: decoder queries attend to encoder states.
        normalized = self.norm_cross_attention(x)
        x = x + self.residual_dropout(
            self.cross_attention(
                normalized, encoder_output, encoder_output,
                key_padding_mask=source_padding_mask,
            )
        )

        # 3) Position-wise feed-forward network.
        x = x + self.residual_dropout(self.ffn(self.norm_ffn(x)))
        return x


@dataclass
class ModelConfig:
    source_vocab_size: int
    target_vocab_size: int
    pad_id: int
    max_len: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id
        self.d_model = config.d_model

        self.source_embedding = nn.Embedding(
            config.source_vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.target_embedding = nn.Embedding(
            config.target_vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.source_position = LearnedPositionalEmbedding(config.max_len, config.d_model)
        self.target_position = LearnedPositionalEmbedding(config.max_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.generator = nn.Linear(config.d_model, config.target_vocab_size)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def embed_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.source_embedding(source_ids) * math.sqrt(self.d_model)
        return self.embedding_dropout(self.source_position(embeddings))

    def embed_target(self, target_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.target_embedding(target_ids) * math.sqrt(self.d_model)
        return self.embedding_dropout(self.target_position(embeddings))

    def encode(self, source_ids: torch.Tensor, source_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed_source(source_ids)
        for layer in self.encoder_layers:
            x = layer(x, source_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        target_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_length = target_ids.size(1)
        # True above the diagonal means a future token is masked.
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, dtype=torch.bool, device=target_ids.device),
            diagonal=1,
        )
        x = self.embed_target(target_ids)
        for layer in self.decoder_layers:
            x = layer(
                x, encoder_output, target_padding_mask,
                source_padding_mask, causal_mask,
            )
        return self.decoder_norm(x)

    def forward(self, source_ids: torch.Tensor, target_input_ids: torch.Tensor) -> torch.Tensor:
        source_padding_mask = source_ids.eq(self.pad_id)
        target_padding_mask = target_input_ids.eq(self.pad_id)
        encoder_output = self.encode(source_ids, source_padding_mask)
        decoder_output = self.decode(
            target_input_ids, encoder_output,
            source_padding_mask, target_padding_mask,
        )
        return self.generator(decoder_output)


# ---------------------------------------------------------------------------
# Training and checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint_atomic(checkpoint: dict, destination: Path) -> None:
    """Write to a temporary file first, then atomically replace the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, destination)


def save_pickle_atomic(value: object, destination: Path) -> None:
    """Write pickle data atomically, just like the model checkpoints."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, destination)


def calculate_epoch_loss(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    pad_id: int,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_tokens = 0

    for source_ids, target_ids in data_loader:
        source_ids = source_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        # Teacher forcing: decoder input is shifted right; labels are shifted left.
        target_input = target_ids[:, :-1]
        target_labels = target_ids[:, 1:]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(source_ids, target_input)
            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_labels.reshape(-1),
            )
            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        token_count = int(target_labels.ne(pad_id).sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot training and validation loss curves across epochs using matplotlib."""
    if not train_losses or not val_losses:
        print("No loss history to plot.")
        return

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, marker="o", markersize=3)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2, marker="s", markersize=3)
    plt.title("Training and Validation Loss vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Loss curve saved to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception as error:
            print(f"Could not display plot window: {error}")
    else:
        plt.close()


def split_pairs(
    pairs: Sequence[Tuple[str, str]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not 0.05 <= val_ratio <= 0.4:
        raise ValueError("--val-ratio must be between 0.05 and 0.4")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_size:], shuffled[:val_size]


def train(args: argparse.Namespace) -> Tuple[Path, Path]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    pairs = read_parallel_corpus(args.data)
    train_pairs, validation_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    # Build vocabularies from training data only to avoid validation leakage.
    source_vocab = Vocabulary.build(source for source, _ in train_pairs)
    target_vocab = Vocabulary.build(target for _, target in train_pairs)

    train_dataset = TranslationDataset(
        train_pairs, source_vocab, target_vocab, args.max_len
    )
    validation_dataset = TranslationDataset(
        validation_pairs, source_vocab, target_vocab, args.max_len
    )

    collate_fn = make_collate_fn(source_vocab.pad_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    config = ModelConfig(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        pad_id=source_vocab.pad_id,
        max_len=args.max_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = TransformerEncoderDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_function = nn.CrossEntropyLoss(
        ignore_index=source_vocab.pad_id,
        label_smoothing=args.label_smoothing,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.checkpoint_dir / "last_model.pt"
    best_path = args.checkpoint_dir / "best_model.pt"
    english_vocabulary_path = args.checkpoint_dir / "english_vocabulary.pkl"
    italian_vocabulary_path = args.checkpoint_dir / "italian_vocabulary.pkl"

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    print(
        f"Sentence pairs: {len(pairs)} | train: {len(train_dataset)} | "
        f"validation: {len(validation_dataset)}"
    )
    print(
        f"Source vocab: {len(source_vocab)} | target vocab: {len(target_vocab)} | "
        f"parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = calculate_epoch_loss(
            model, train_loader, loss_function, device, source_vocab.pad_id,
            optimizer=optimizer, grad_clip=args.grad_clip,
        )
        validation_loss = calculate_epoch_loss(
            model, validation_loader, loss_function, device, source_vocab.pad_id
        )
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "current_validation_loss": validation_loss,
            "train_losses": list(train_losses),
            "val_losses": list(val_losses),
            "model_config": asdict(config),
            "source_vocab": source_vocab.itos,
            "target_vocab": target_vocab.itos,
            "training_config": {
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "max_len": args.max_len,
            },
        }

        # Always replace last_model.pt at the end of every completed epoch.
        save_checkpoint_atomic(checkpoint, last_path)
        save_pickle_atomic(source_vocab.itos, english_vocabulary_path)
        save_pickle_atomic(target_vocab.itos, italian_vocabulary_path)
        # Replace best_model.pt only when validation loss improves.
        if improved:
            save_checkpoint_atomic(checkpoint, best_path)
            best_marker = " | new best"
        else:
            best_marker = ""

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"best_val={best_validation_loss:.4f} | lr={current_lr:.2e}"
            f"{best_marker}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping: validation loss did not improve for "
                f"{args.early_stopping_patience} consecutive epochs."
            )
            break

    print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"English vocabulary: {english_vocabulary_path}")
    print(f"Italian vocabulary: {italian_vocabulary_path}")

    loss_plot_path = args.checkpoint_dir / "loss_curve.png"
    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=loss_plot_path,
        show=not getattr(args, "no_plot", False),
    )

    return last_path, best_path


# ---------------------------------------------------------------------------
# Loading and interactive greedy decoding
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is appropriate here because this is our own checkpoint
    # and it contains optimizer/config metadata in addition to tensor weights.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)

    source_vocab = Vocabulary(checkpoint["source_vocab"])
    target_vocab = Vocabulary(checkpoint["target_vocab"])
    config = ModelConfig(**checkpoint["model_config"])
    model = TransformerEncoderDecoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, source_vocab, target_vocab, checkpoint


@torch.inference_mode()
def translate(
    text: str,
    model: TransformerEncoderDecoder,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    device: torch.device,
    max_decode_len: int | None = None,
) -> str:
    source_tokens = source_vocab.encode(text)
    if not source_tokens:
        return "Please enter a non-empty English sentence."

    max_len = max_decode_len or model.config.max_len
    source_ids = torch.tensor(
        source_tokens + [source_vocab.eos_id], dtype=torch.long, device=device
    ).unsqueeze(0)
    if source_ids.size(1) > model.config.max_len:
        raise ValueError(
            f"Input is too long ({source_ids.size(1)} tokens). "
            f"Maximum supported length is {model.config.max_len}."
        )

    source_padding_mask = source_ids.eq(source_vocab.pad_id)
    encoder_output = model.encode(source_ids, source_padding_mask)

    generated = [target_vocab.bos_id]
    for _ in range(max_len - 1):
        target_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        target_padding_mask = target_ids.eq(target_vocab.pad_id)
        decoder_output = model.decode(
            target_ids,
            encoder_output,
            source_padding_mask,
            target_padding_mask,
        )
        next_token_logits = model.generator(decoder_output[:, -1, :])
        # These special tokens are not valid output words.
        next_token_logits[:, target_vocab.pad_id] = float("-inf")
        next_token_logits[:, target_vocab.bos_id] = float("-inf")
        next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

        if next_token_id == target_vocab.eos_id:
            break
        generated.append(next_token_id)

    output_tokens = [
        target_vocab.itos[token_id]
        for token_id in generated[1:]
        if token_id not in {target_vocab.pad_id, target_vocab.bos_id, target_vocab.eos_id}
    ]
    return detokenize(output_tokens) if output_tokens else "(no translation generated)"


def interactive_translation(
    model: TransformerEncoderDecoder,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    device: torch.device,
) -> None:
    print("\nInteractive English -> Italian translation")
    print("Enter an English sentence. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            sentence = input("English> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting translation mode.")
            break
        if sentence.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not sentence:
            continue
        try:
            result = translate(sentence, model, source_vocab, target_vocab, device)
            print(f"Italian> {result}\n")
        except ValueError as error:
            print(f"Error: {error}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a word-level English-to-Italian Transformer encoder-decoder."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path to a CSV/TSV corpus with english and italian columns",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2, help="Encoder and decoder layers")
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Stop after this many epochs without improvement; 0 disables it")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no-plot", action="store_true", help="Do not display the matplotlib loss curve window")
    parser.add_argument("--skip-interactive", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.num_heads < 1:
        parser.error("--num-heads must be at least 1")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if args.layers < 1 or args.ff_dim < 1:
        parser.error("--layers and --ff-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in the range [0, 1)")
    if args.max_len < 4:
        parser.error("--max-len must be at least 4")
    return args


def main() -> None:
    args = parse_args()
    last_checkpoint_path, _ = train(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("\nLoading last_model.pt for manual testing...")
    model, source_vocab, target_vocab, checkpoint = load_checkpoint(last_checkpoint_path, device)
    print(
        f"Loaded epoch {checkpoint['epoch']} | "
        f"validation loss: {checkpoint['current_validation_loss']:.4f} | "
        f"device: {device}"
    )

    if not args.skip_interactive:
        interactive_translation(model, source_vocab, target_vocab, device)


if __name__ == "__main__":
    main()
