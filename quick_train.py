"""
quick_train.py — Validates that the model genuinely learns on real WikiText-2,
                 but completes in ~15 minutes on M4 MacBook Air.

What this proves:
  • Data pipeline works (tokeniser, chunking, batching)
  • Model + diffusion + loss work end-to-end on real text
  • Loss decreases on real data (not just a synthetic batch)
  • Generation produces readable text fragments

What it doesn't prove:
  • Final generation quality (need full 50k steps for that)

After this succeeds, run:
    python src/train.py    ← full run, ~1.5 hrs on M4 (see note below)

Usage:
    cd sddlm/
    python quick_train.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, ModelConfig, DiffusionConfig, TrainingConfig, LossConfig
from src.dataset import get_dataloaders
from src.model import DiffusionLM
from src.diffusion import NoiseSchedule, UniformDiffusion
from src.loss import compute_loss

# ─────────────────────────────────────────────────────────────────────────────
# Config: same architecture as the full run, fewer steps
# Keep d_model / n_layers IDENTICAL to full config so you're testing
# the real model, not a toy.
# ─────────────────────────────────────────────────────────────────────────────

cfg = Config()

# Model: same as full training
cfg.model = ModelConfig(
    vocab_size=50257,
    d_model=256,
    n_heads=4,
    n_layers=6,
    d_ff=1024,
    dropout=0.1,
    max_seq_len=128,
)

cfg.diffusion = DiffusionConfig(
    num_timesteps=1000,
    schedule="cosine",
    eps=1e-4,
)

# Training: 3000 steps ≈ 15 min on M4
cfg.training = TrainingConfig(
    data_dir="data/wikitext2",
    batch_size=16,
    learning_rate=3e-4,
    warmup_steps=200,  # shorter warmup for quick run
    max_steps=3000,
    eval_every=300,
    save_every=3000,  # only save at the end
    checkpoint_dir="checkpoints/quick",
    device="auto",
    weight_decay=0.0,
    grad_clip=1.0,
    adam_beta2=0.999,
)

cfg.loss = LossConfig(
    loss_type="sddlm_v1",
    epsilon=1e-6,
    n_neg_samples=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(step, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


@torch.no_grad()
def eval_loss(model, diffusion, sched, loader, loss_cfg, device, n_batches=30):
    model.eval()
    total = 0.0
    for i, x0 in enumerate(loader):
        if i >= n_batches:
            break
        x0 = x0.to(device)
        t = sched.sample_t(x0.shape[0], device)
        xt, _ = diffusion.q_sample(x0, t)
        logits = model(xt, sched.t_to_float(t))
        loss, _ = compute_loss(logits, x0, xt, loss_cfg)
        total += loss.item()
    model.train()
    return total / min(n_batches, i + 1)


def decode_sample(ids, tokenizer):
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Quick training run — 3000 steps")
    print(f"  device : {device}")
    print(f"  Expected time : ~15 min on M4, ~5 min on CUDA")
    print(f"{'='*60}\n")

    if not os.path.exists(os.path.join(cfg.training.data_dir, "train.txt")):
        print("ERROR: data/wikitext2/train.txt not found.")
        print("Run the download snippet from README.md first.")
        sys.exit(1)

    # ── data ────────────────────────────────────────────────────────────────
    train_loader, test_loader, tokenizer = get_dataloaders(cfg)

    # ── model ────────────────────────────────────────────────────────────────
    model = DiffusionLM(cfg.model).to(device)

    # ── diffusion ────────────────────────────────────────────────────────────
    sched = NoiseSchedule(
        cfg.diffusion.num_timesteps, cfg.diffusion.schedule, cfg.diffusion.eps
    )
    diffusion = UniformDiffusion(sched, cfg.model.vocab_size)

    # ── optimiser ────────────────────────────────────────────────────────────
    opt = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, cfg.training.adam_beta2),
        weight_decay=cfg.training.weight_decay,
    )

    # ── training ─────────────────────────────────────────────────────────────
    model.train()
    train_iter = iter(train_loader)
    ema_loss = None
    t0 = time.time()
    loss_log = []

    print(f"{'Step':>7}  {'Loss':>8}  {'EMA':>8}  {'Frac':>6}  {'Steps/s':>8}")
    print("─" * 50)

    for step in range(1, cfg.training.max_steps + 1):
        # fetch batch
        try:
            x0 = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x0 = next(train_iter)
        x0 = x0.to(device)

        # lr schedule
        lr = get_lr(step, cfg.training.warmup_steps, cfg.training.learning_rate)
        for g in opt.param_groups:
            g["lr"] = lr

        # forward diffusion
        t_idx = sched.sample_t(x0.shape[0], device)
        xt, _ = diffusion.q_sample(x0, t_idx)
        t_float = sched.t_to_float(t_idx)

        # model + loss
        logits = model(xt, t_float)
        loss, info = compute_loss(logits, x0, xt, cfg.loss)

        # backward
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()

        lv = loss.item()
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
        loss_log.append(lv)

        if step % 100 == 0:
            elapsed = time.time() - t0
            sps = step / elapsed
            eta_min = (cfg.training.max_steps - step) / sps / 60
            print(
                f"{step:>7}  {lv:>8.4f}  {ema_loss:>8.4f}  "
                f"{info['frac_corrupted']:>6.2f}  {sps:>8.1f}  "
                f"(ETA {eta_min:.0f} min)"
            )

        # validation
        if step % cfg.training.eval_every == 0:
            val_loss = eval_loss(model, diffusion, sched, test_loader, cfg.loss, device)
            print(f"  ↳ [val] loss={val_loss:.4f}")

    # ── save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "quick_3k.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved → {ckpt_path}")

    # ── loss sanity check ─────────────────────────────────────────────────────
    first100 = sum(loss_log[:100]) / 100
    last100 = sum(loss_log[-100:]) / 100
    dropped = (first100 - last100) / first100 * 100
    print(f"\nLoss: {first100:.4f} → {last100:.4f}  ({dropped:.1f}% drop)")

    if dropped < 5:
        print("⚠  Loss barely moved. Check data path and try again.")
    else:
        print("✓  Loss is decreasing — model is learning correctly.")

    # ── generate a few samples ────────────────────────────────────────────────
    print("\n── Sample outputs (64 denoising steps) ──")
    model.eval()
    with torch.no_grad():
        x_gen = diffusion.sample(
            model, batch_size=2, seq_len=64, num_steps=64, device=device
        )
    for i, ids in enumerate(x_gen):
        text = decode_sample(ids, tokenizer)
        print(f"\n  [{i+1}] {text[:200]}")

    total_time = (time.time() - t0) / 60
    print(f"\nTotal time: {total_time:.1f} min")
    print("\nIf samples look like word fragments / partial sentences → CORRECT.")
    print("At 3000 steps the model is not converged — that's expected.")
    print("To fully train: python src/train.py")


if __name__ == "__main__":
    main()
