"""
train.py — Main training loop for SDDLM / SDDLM-V1.

Usage:
    cd sddlm/
    python src/train.py

Key features:
  • Auto-detects device: MPS (Apple Silicon) > CUDA > CPU
  • Linear LR warmup followed by constant LR (paper setting for small models)
  • Gradient clipping (prevents explosion, common in diffusion training)
  • Periodic validation loss + checkpoint saving
  • Resumes from latest checkpoint if one exists
"""

import os
import sys
import time
import math
import glob

import torch
import torch.nn as nn
from torch.optim import AdamW

# ── allow running as  python src/train.py  from the project root ──────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, ModelConfig, DiffusionConfig, TrainingConfig, LossConfig
from src.dataset import get_dataloaders
from src.model import DiffusionLM
from src.diffusion import NoiseSchedule, UniformDiffusion
from src.loss import compute_loss

# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """
    Linear warmup for the first `warmup_steps`, then constant.

    Why warmup?
      Early in training the gradients are noisy and large.
      Starting with a small LR prevents the first few batches from
      blowing up the freshly-initialised weights.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic rename — safe against crash mid-write
    print(f"  [ckpt] saved → {path}")


def load_latest_checkpoint(ckpt_dir: str, model, optimizer):
    """Load the most recent checkpoint if any. Returns start_step."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if not files:
        return 0
    path = files[-1]
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]
    print(f"  [ckpt] resumed from {path}  (step {step})")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, diffusion, test_loader, loss_cfg, device, max_batches=50):
    """
    Compute average validation loss over up to `max_batches` batches.
    Uses the same loss function as training.
    """
    model.eval()
    total_loss = 0.0
    total_frac = 0.0
    n_batches = 0

    for x0 in test_loader:
        if n_batches >= max_batches:
            break
        x0 = x0.to(device)
        B = x0.shape[0]

        t_idx = diffusion.schedule.sample_t(B, device)
        xt, _ = diffusion.q_sample(x0, t_idx)
        t_float = diffusion.schedule.t_to_float(t_idx)
        logits = model(xt, t_float)
        loss, info = compute_loss(logits, x0, xt, loss_cfg)

        total_loss += loss.item()
        total_frac += info["frac_corrupted"]
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1), total_frac / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(config: Config):
    # ── setup ──────────────────────────────────────────────────────────────
    device = get_device(config.training.device)
    print(f"\n{'='*60}")
    print(f"  SDDLM Training")
    print(f"  device     : {device}")
    print(f"  loss       : {config.loss.loss_type}")
    print(f"  batch_size : {config.training.batch_size}")
    print(f"  seq_len    : {config.model.max_seq_len}")
    print(f"  max_steps  : {config.training.max_steps}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, test_loader, tokenizer = get_dataloaders(config)

    # ── model ─────────────────────────────────────────────────────────────
    model = DiffusionLM(config.model).to(device)

    # ── diffusion ─────────────────────────────────────────────────────────
    schedule = NoiseSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        schedule=config.diffusion.schedule,
        eps=config.diffusion.eps,
    )
    diffusion = UniformDiffusion(schedule, vocab_size=config.model.vocab_size)

    # ── optimiser ─────────────────────────────────────────────────────────
    # Separate weight-decay and no-decay parameter groups:
    # Biases, layer-norm params, and embeddings should NOT have weight decay.
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name or "bias" in name or "emb" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(0.9, config.training.adam_beta2),
        eps=1e-8,
    )

    # ── resume from checkpoint if available ───────────────────────────────
    start_step = load_latest_checkpoint(
        config.training.checkpoint_dir, model, optimizer
    )

    # ── training ──────────────────────────────────────────────────────────
    model.train()
    step = start_step
    train_iter = iter(train_loader)

    # Exponential moving average of loss (for smoother logging)
    ema_loss = None
    t0 = time.time()

    while step < config.training.max_steps:
        # ── fetch batch (cycle through dataset) ────────────────────────────
        try:
            x0 = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x0 = next(train_iter)
        x0 = x0.to(device)  # (B, L)
        B = x0.shape[0]

        # ── update learning rate ────────────────────────────────────────────
        lr = get_lr(step, config.training.warmup_steps, config.training.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── forward diffusion: x_0 → x_t ───────────────────────────────────
        t_idx = schedule.sample_t(B, device)  # (B,) random timesteps
        xt, _ = diffusion.q_sample(x0, t_idx)  # (B, L) corrupted
        t_float = schedule.t_to_float(t_idx)  # (B,) fractions

        # ── model forward pass ──────────────────────────────────────────────
        logits = model(xt, t_float)  # (B, L, V)

        # ── loss ────────────────────────────────────────────────────────────
        loss, info = compute_loss(logits, x0, xt, config.loss)

        # ── backward ────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        # Very important for SDDLM-V1 because the negative-gradient term
        # can produce large gradients when log p(x̂) → -∞ for rare tokens
        nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        optimizer.step()
        step += 1

        # ── logging ─────────────────────────────────────────────────────────
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val

        if step % 100 == 0:
            elapsed = time.time() - t0
            steps_ps = step / elapsed
            eta_min = (config.training.max_steps - step) / steps_ps / 60
            print(
                f"step {step:6d}/{config.training.max_steps} | "
                f"loss {loss_val:.4f} (ema {ema_loss:.4f}) | "
                f"frac_corrupt {info['frac_corrupted']:.2f} | "
                f"lr {lr:.2e} | "
                f"steps/s {steps_ps:.1f} | "
                f"ETA {eta_min:.0f} min"
            )

        # ── validation ──────────────────────────────────────────────────────
        if step % config.training.eval_every == 0:
            val_loss, val_frac = evaluate(
                model, diffusion, test_loader, config.loss, device
            )
            print(
                f"  [val] step {step} | "
                f"val_loss {val_loss:.4f} | "
                f"frac_corrupt {val_frac:.2f}"
            )

        # ── checkpoint ──────────────────────────────────────────────────────
        if step % config.training.save_every == 0:
            save_checkpoint(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(config.training.checkpoint_dir, f"step_{step:07d}.pt"),
            )

    # ── final checkpoint ────────────────────────────────────────────────────
    save_checkpoint(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        os.path.join(config.training.checkpoint_dir, "final.pt"),
    )
    print(f"\nTraining complete  ({step} steps)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()

    # ── Tiny override for quick smoke-test on CPU ──────────────────────────
    # Remove or comment out these lines for full training
    # cfg.model.n_layers    = 2
    # cfg.model.d_model     = 64
    # cfg.model.d_ff        = 256
    # cfg.model.n_heads     = 2
    # cfg.training.max_steps = 500
    # cfg.training.eval_every = 100

    train(cfg)
