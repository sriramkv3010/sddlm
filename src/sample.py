"""
sample.py — Generate text from a trained SDDLM checkpoint.

Usage:
    cd sddlm/
    python src/sample.py --checkpoint checkpoints/final.pt --steps 64 --n 4

Generation process:
  1. Start: x_T ~ Uniform(V)  — pure noise (random tokens)
  2. For t = T-1, T-2, ..., 1:
       a. Feed (x_t, t) to model → p_θ(x_0 | x_t)
       b. Sample x_0_hat
       c. Re-corrupt to level (t-1): x_{t-1} ~ q(x_{t-1} | x_0_hat)
  3. Final: argmax of model output at t=0

The quality improves with more sampling steps (up to a point).
On WikiText-2 scale, 64–256 steps usually suffices.
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import DiffusionLM
from src.diffusion import NoiseSchedule, UniformDiffusion
from src.dataset import get_tokenizer


def load_model(ckpt_path: str, device: torch.device):
    """Load model and config from a checkpoint file."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = DiffusionLM(cfg.model).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}  (step {state['step']})")
    return model, cfg


def generate(
    model,
    diffusion: UniformDiffusion,
    tokenizer,
    n_sequences: int = 4,
    seq_len: int = 128,
    num_steps: int = 64,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """
    Generate `n_sequences` text samples.

    Returns:
        List of decoded strings.
    """
    token_ids = diffusion.sample(
        model=model,
        batch_size=n_sequences,
        seq_len=seq_len,
        num_steps=num_steps,
        device=device,
    )  # (n, seq_len)

    texts = []
    for ids in token_ids:
        # Skip special tokens (EOS=50256) when decoding
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained SDDLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final.pt",
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of denoising steps (more = better quality, slower)",
    )
    parser.add_argument(
        "--n", type=int, default=4, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--seq_len", type=int, default=128, help="Sequence length in tokens"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto / cpu / mps / cuda"
    )
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at '{args.checkpoint}'")
        print("Train first:  python src/train.py")
        sys.exit(1)

    model, cfg = load_model(args.checkpoint, device)

    # ── diffusion ─────────────────────────────────────────────────────────
    schedule = NoiseSchedule(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.schedule,
        eps=cfg.diffusion.eps,
    )
    diffusion = UniformDiffusion(schedule, vocab_size=cfg.model.vocab_size)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = get_tokenizer()

    # ── generate ──────────────────────────────────────────────────────────
    print(f"\nGenerating {args.n} sequences  ({args.steps} denoising steps)…\n")
    texts = generate(
        model,
        diffusion,
        tokenizer,
        n_sequences=args.n,
        seq_len=args.seq_len,
        num_steps=args.steps,
        device=device,
    )

    for i, text in enumerate(texts):
        print(f"{'─'*60}")
        print(f"Sample {i+1}:")
        print(text)
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
