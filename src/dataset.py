"""WikiText-103 dataset pipeline for distributed pretraining.

Handles:
1. Automated downloading of WikiText-103 raw dataset from the authoritative URL:
   https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true
2. High-performance GPT-2 standard BPE tokenization via tiktoken (vocab_size = 50,257).
3. Memory-mapped binary token storage for zero-overhead streaming during training.
4. Deterministic SPMD data-parallel batch sharding across TPU hosts and devices.
"""

from pathlib import Path
import urllib.request
import zipfile
from typing import Generator, Optional, Tuple

import numpy as np
import tiktoken

WIKITEXT_URL = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def ensure_wikitext103_ready(data_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Ensures tokenized WikiText-103 binary arrays exist, downloading if necessary.

    Returns:
        Tuple of (train_npy_path, valid_npy_path).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_npy = data_dir / "train.npy"
    valid_npy = data_dir / "valid.npy"

    if train_npy.exists() and valid_npy.exists():
        return train_npy, valid_npy

    raw_dir = data_dir / "wikitext-103-raw"
    train_raw = raw_dir / "wiki.train.raw"
    valid_raw = raw_dir / "wiki.valid.raw"

    if not (train_raw.exists() and valid_raw.exists()):
        zip_path = data_dir / "wikitext-103-raw-v1.zip"
        if not zip_path.exists():
            print(f"Downloading WikiText-103 from {WIKITEXT_URL}...")
            urllib.request.urlretrieve(WIKITEXT_URL, zip_path)
        print("Extracting WikiText-103...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    enc = tiktoken.get_encoding("gpt2")
    if not valid_npy.exists():
        print("Tokenizing wiki.valid.raw...")
        valid_text = valid_raw.read_text(encoding="utf-8")
        valid_tokens = np.array(enc.encode(valid_text), dtype=np.uint16)
        np.save(valid_npy, valid_tokens)

    if not train_npy.exists():
        print("Tokenizing wiki.train.raw...")
        train_text = train_raw.read_text(encoding="utf-8")
        train_tokens = np.array(enc.encode(train_text), dtype=np.uint16)
        np.save(train_npy, train_tokens)

    return train_npy, valid_npy


class ShardedTokenLoader:
    """SPMD data-parallel sequence batch generator for distributed pretraining."""

    def __init__(
        self,
        token_path: Path,
        batch_size: int = 64,
        seq_len: int = 512,
        process_index: int = 0,
        process_count: int = 1,
        seed: int = 42,
    ):
        self.tokens = np.load(token_path, mmap_mode="r")
        self.total_tokens = len(self.tokens)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count

        assert batch_size % process_count == 0, (
            f"batch_size ({batch_size}) must be divisible by process_count ({process_count})"
        )
        self.local_batch_size = batch_size // process_count
        self.span = seq_len + 1  # tokens + target shifted by 1
        self.num_possible_sequences = (self.total_tokens - 1) // seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_possible_sequences // self.batch_size

    def get_batch(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates process-local slice of the global batch for the given step.

        Returns:
            x: input token ids of shape (local_batch_size, seq_len)
            y: target token ids of shape (local_batch_size, seq_len)
        """
        # Deterministic sequence offset calculation across all hosts
        step_rng = np.random.default_rng(10007 * (step + 1) + 42)
        global_indices = step_rng.integers(
            0, self.total_tokens - self.span, size=self.batch_size
        )

        # Slice the process-local chunk
        start_idx = self.process_index * self.local_batch_size
        end_idx = start_idx + self.local_batch_size
        local_indices = global_indices[start_idx:end_idx]

        x_batch = np.empty((self.local_batch_size, self.seq_len), dtype=np.int32)
        y_batch = np.empty((self.local_batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(local_indices):
            chunk = self.tokens[idx : idx + self.span].astype(np.int32)
            x_batch[i] = chunk[:self.seq_len]
            y_batch[i] = chunk[1 : self.span]

        return x_batch, y_batch


def evaluate_perplexity(
    model,
    params,
    valid_tokens_path: Path,
    seq_len: int = 512,
    batch_size: int = 32,
    max_eval_batches: int = 20,
    is_algebraic: bool = True,
) -> float:
    """Evaluates held-out validation perplexity on WikiText-103."""
    tokens = np.load(valid_tokens_path, mmap_mode="r")
    total_tokens = len(tokens)
    span = seq_len + 1

    total_nll = 0.0
    total_count = 0

    num_seqs = min((total_tokens - 1) // seq_len, max_eval_batches * batch_size)
    num_batches = num_seqs // batch_size

    for b in range(num_batches):
        batch_starts = [i * seq_len for i in range(b * batch_size, (b + 1) * batch_size)]
        x = np.stack([tokens[s : s + seq_len].astype(np.int32) for s in batch_starts])
        y = np.stack([tokens[s + 1 : s + span].astype(np.int32) for s in batch_starts])

        logits = model.forward(params, x)
        if is_algebraic:
            # Octic A-Softmax probability evaluation on closed vocabulary simplex (zero sink)
            from src.attention import algebraic_softmax
            eps_v = float(getattr(model.config, "eps_vocab", 100.0))
            probs = np.asarray(algebraic_softmax(logits, sink_omega=0.0, eps=eps_v))
            # Clamp to prevent log(0)
            probs = np.maximum(probs, 1e-12)
            # Pick target probabilities
            target_probs = np.take_along_axis(probs, y[..., None], axis=-1).squeeze(-1)
            batch_nll = -np.sum(np.log(target_probs))
        else:
            import jax
            log_probs = np.asarray(jax.nn.log_softmax(logits, axis=-1))
            target_log_probs = np.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
            batch_nll = -np.sum(target_log_probs)

        total_nll += float(batch_nll)
        total_count += x.size

    avg_nll = total_nll / total_count
    return float(np.exp(avg_nll))
