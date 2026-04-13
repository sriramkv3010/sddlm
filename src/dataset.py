"""
dataset.py — WikiText-2 loading, tokenisation, and batching.

Design decisions:
  - Non-overlapping fixed-length chunks (simplest, no data leakage)
  - GPT-2 tokenizer (50257 tokens, byte-level BPE)
  - num_workers=0 — required for MPS (Apple Silicon) compatibility
  - pre-tokenise once and store in memory (WikiText-2 fits easily in 16 GB)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class TextChunkDataset(Dataset):
    """
    Reads a plain-text file, tokenises the whole thing, then serves
    non-overlapping windows of length `seq_len`.

    Why non-overlapping?
      - Overlapping windows create correlated batches which inflate effective
        dataset size without adding new signal.
      - Diffusion models don't need autoregressive left-to-right ordering,
        so random access over chunks is perfectly fine.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
    ):
        self.seq_len = seq_len

        # ── load raw text ──────────────────────────────────────────────────
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # WikiText-2 uses " @-@ " for hyphens, " @,@ " for commas, etc.
        # We keep them as-is; the BPE tokenizer handles them fine.

        # ── tokenise ───────────────────────────────────────────────────────
        # GPT-2 tokenizer returns a plain Python list of ints
        token_ids = tokenizer.encode(text)
        all_ids = torch.tensor(token_ids, dtype=torch.long)

        # ── chunk into fixed-length pieces ─────────────────────────────────
        n_chunks = len(all_ids) // seq_len
        self.data = all_ids[: n_chunks * seq_len].view(n_chunks, seq_len)

        print(
            f"[{split}] {file_path} → "
            f"{len(all_ids):,} tokens → "
            f"{n_chunks:,} chunks × {seq_len} tokens"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns a single chunk: (seq_len,) LongTensor
        return self.data[idx]


def get_tokenizer():
    """
    Returns a GPT-2 tokenizer from HuggingFace Transformers.

    Why GPT-2 tokenizer?
      - 50 257-token vocabulary — large enough to capture English subwords
      - Byte-level BPE: no unknown tokens, handles any Unicode
      - Same tokenizer used by the paper (Llama-2 for large experiments,
        but GPT-2 is equivalent for WikiText-2 scale)
    """
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 has no pad token; map it to EOS so padding doesn't crash
    tok.pad_token = tok.eos_token
    return tok


def get_dataloaders(config):
    """
    Build train and test DataLoaders.

    Returns:
        train_loader, test_loader, tokenizer
    """
    tokenizer = get_tokenizer()

    train_ds = TextChunkDataset(
        file_path=os.path.join(config.training.data_dir, "train.txt"),
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        split="train",
    )
    test_ds = TextChunkDataset(
        file_path=os.path.join(config.training.data_dir, "test.txt"),
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # MPS requires num_workers=0
        pin_memory=False,
        drop_last=True,  # keeps batch size constant → cleaner loss curves
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, test_loader, tokenizer
    """
dataset.py — WikiText-2 loading, tokenisation, and batching.

Design decisions:
  - Non-overlapping fixed-length chunks (simplest, no data leakage)
  - GPT-2 tokenizer (50257 tokens, byte-level BPE)
  - num_workers=0 — required for MPS (Apple Silicon) compatibility
  - pre-tokenise once and store in memory (WikiText-2 fits easily in 16 GB)
"""


import os
import torch
from torch.utils.data import Dataset, DataLoader


class TextChunkDataset(Dataset):
    """
    Reads a plain-text file, tokenises the whole thing, then serves
    non-overlapping windows of length `seq_len`.

    Why non-overlapping?
      - Overlapping windows create correlated batches which inflate effective
        dataset size without adding new signal.
      - Diffusion models don't need autoregressive left-to-right ordering,
        so random access over chunks is perfectly fine.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
    ):
        self.seq_len = seq_len

        # ── load raw text ──────────────────────────────────────────────────
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # WikiText-2 uses " @-@ " for hyphens, " @,@ " for commas, etc.
        # We keep them as-is; the BPE tokenizer handles them fine.

        # ── tokenise ───────────────────────────────────────────────────────
        # GPT-2 tokenizer returns a plain Python list of ints
        token_ids = tokenizer.encode(text)
        all_ids = torch.tensor(token_ids, dtype=torch.long)

        # ── chunk into fixed-length pieces ─────────────────────────────────
        n_chunks = len(all_ids) // seq_len
        self.data = all_ids[: n_chunks * seq_len].view(n_chunks, seq_len)

        print(
            f"[{split}] {file_path} → "
            f"{len(all_ids):,} tokens → "
            f"{n_chunks:,} chunks × {seq_len} tokens"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns a single chunk: (seq_len,) LongTensor
        return self.data[idx]


def get_tokenizer():
    """
    Returns a GPT-2 tokenizer from HuggingFace Transformers.

    Why GPT-2 tokenizer?
      - 50 257-token vocabulary — large enough to capture English subwords
      - Byte-level BPE: no unknown tokens, handles any Unicode
      - Same tokenizer used by the paper (Llama-2 for large experiments,
        but GPT-2 is equivalent for WikiText-2 scale)
    """
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 has no pad token; map it to EOS so padding doesn't crash
    tok.pad_token = tok.eos_token
    return tok


def get_dataloaders(config):
    """
    Build train and test DataLoaders.

    Returns:
        train_loader, test_loader, tokenizer
    """
    tokenizer = get_tokenizer()

    train_ds = TextChunkDataset(
        file_path=os.path.join(config.training.data_dir, "train.txt"),
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        split="train",
    )
    test_ds = TextChunkDataset(
        file_path=os.path.join(config.training.data_dir, "test.txt"),
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # MPS requires num_workers=0
        pin_memory=False,
        drop_last=True,  # keeps batch size constant → cleaner loss curves
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, test_loader, tokenizer
