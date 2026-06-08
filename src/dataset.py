import os
import torch
from torch.utils.data import Dataset, DataLoader


class TextChunkDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
    ):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        all_ids = torch.tensor(token_ids, dtype=torch.long)
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
        return self.data[idx]


def get_tokenizer():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def get_dataloaders(config):
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
        num_workers=0,  
        pin_memory=False,
        drop_last=True, 
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
   
import os
import torch
from torch.utils.data import Dataset, DataLoader


class TextChunkDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 128,
        split: str = "train",
    ):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        all_ids = torch.tensor(token_ids, dtype=torch.long)
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
        return self.data[idx]


def get_tokenizer():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 has no pad token; map it to EOS so padding doesn't crash
    tok.pad_token = tok.eos_token
    return tok


def get_dataloaders(config):
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
        num_workers=0,  
        pin_memory=False,
        drop_last=True,  
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
