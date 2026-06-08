import sys
import os
import math
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, ModelConfig, DiffusionConfig, LossConfig
from src.model import DiffusionLM
from src.diffusion import NoiseSchedule, UniformDiffusion
from src.loss import sddlm_loss, sddlm_v1_loss, compute_loss

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[94m{}\033[0m"


def section(name):
    print(f"\n{HEAD.format('─'*50)}")
    print(f"{HEAD.format(name)}")
    print(f"{HEAD.format('─'*50)}")


def check(condition: bool, msg: str, extra: str = ""):
    tag = PASS if condition else FAIL
    print(f"  {tag}  {msg}" + (f"  [{extra}]" if extra else ""))
    if not condition:
        raise AssertionError(f"FAILED: {msg}")


def no_nan_inf(tensor: torch.Tensor, name: str):
    bad = torch.isnan(tensor).any() or torch.isinf(tensor).any()
    check(not bad, f"{name} has no NaN/Inf")

V = 200  # tiny vocabulary
L = 32  # short sequences
B = 8  # small batch
T = 100  # few diffusion timesteps

cfg = Config()
cfg.model = ModelConfig(
    vocab_size=V,
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=128,
    dropout=0.0,  # off for reproducibility
    max_seq_len=L,
)
cfg.diffusion = DiffusionConfig(num_timesteps=T, schedule="cosine", eps=1e-4)
cfg.loss = LossConfig(loss_type="sddlm_v1", epsilon=1e-6, n_neg_samples=1)

torch.manual_seed(42)
device = torch.device("cpu")  # smoke test always runs on CPU

section("TEST 1 · NoiseSchedule")

sched = NoiseSchedule(T, schedule="cosine", eps=1e-4)

check(sched.alphas.shape == (T,), f"alphas shape is ({T},)", str(sched.alphas.shape))

check(bool((sched.alphas >= 1e-4).all()), "all alphas >= eps (no log-of-zero risk)")

check(bool((sched.alphas <= 1.0).all()), "all alphas <= 1.0")


diffs = sched.alphas[1:] - sched.alphas[:-1]
check(
    bool((diffs <= 1e-5).all()),
    "alphas are monotone non-increasing",
    f"max_increase={diffs.max().item():.2e}",
)

check(sched.alphas[0].item() > 0.98, f"alpha_0 ≈ 1 (got {sched.alphas[0].item():.4f})")

check(
    sched.alphas[-1].item() < 0.01, f"alpha_T ≈ 0 (got {sched.alphas[-1].item():.6f})"
)


t_idx = torch.tensor([0, T // 2, T - 1])
a = sched.get_alpha(t_idx)
check(a.shape == (3,), "get_alpha returns right shape")
check(
    a[0].item() > a[1].item() > a[2].item(),
    "get_alpha preserves ordering (a[0] > a[T/2] > a[T-1])",
)

section("TEST 2 · Forward diffusion  q(x_t | x_0)")

diffusion = UniformDiffusion(sched, vocab_size=V)
x0 = torch.randint(0, V, (B, L))


t_low = torch.zeros(B, dtype=torch.long)
xt_low, alpha_low = diffusion.q_sample(x0, t_low)
frac_changed_low = (xt_low != x0).float().mean().item()
check(
    frac_changed_low < 0.05, f"at t=0, <5% tokens change (got {frac_changed_low:.3f})"
)

t_high = torch.full((B,), T - 1, dtype=torch.long)
xt_high, alpha_high = diffusion.q_sample(x0, t_high)
frac_changed_high = (xt_high != x0).float().mean().item()
check(
    frac_changed_high > 0.90,
    f"at t=T-1, >90% tokens change (got {frac_changed_high:.3f})",
)

t_mid = torch.full((B,), T // 2, dtype=torch.long)
alpha_mid = sched.get_alpha(t_mid)[0].item()
expected_frac = 1.0 - alpha_mid

frac_estimates = []
for _ in range(50):
    xt_m, _ = diffusion.q_sample(x0, t_mid)
    frac_estimates.append((xt_m != x0).float().mean().item())
actual_frac = sum(frac_estimates) / len(frac_estimates)

check(
    abs(actual_frac - expected_frac) < 0.05,
    f"corruption rate ≈ 1-alpha (expected {expected_frac:.3f}, got {actual_frac:.3f})",
)

check(xt_low.shape == x0.shape, "xt has same shape as x0")
check(xt_low.dtype == torch.long, "xt is LongTensor")
check(
    bool((xt_low >= 0).all() and (xt_low < V).all()),
    "all xt tokens are valid vocab indices",
)


section("TEST 3 · SDDLM loss  (Eq. 7)")

model = DiffusionLM(cfg.model)
model.eval()

t_mid = torch.full((B,), T // 2, dtype=torch.long)
xt, _ = diffusion.q_sample(x0, t_mid)
t_f = sched.t_to_float(t_mid)
logits = model(xt, t_f)

# Shape
check(logits.shape == (B, L, V), f"logits shape is ({B},{L},{V})", str(logits.shape))

# When nothing is corrupted, loss must be exactly 0
fake_xt_same = x0.clone()  # xt == x0 everywhere → mask is all zeros
loss_zero, info_z = sddlm_loss(logits, x0, fake_xt_same)
check(
    loss_zero.item() == 0.0,
    f"loss=0 when no tokens corrupted (got {loss_zero.item():.6f})",
)
check(info_z["frac_corrupted"] == 0.0, "frac_corrupted=0 matches")

loss_pos, info_p = sddlm_loss(logits, x0, xt)
check(
    loss_pos.item() > 0.0, f"loss>0 when tokens corrupted (got {loss_pos.item():.4f})"
)

# Loss is finite
no_nan_inf(loss_pos, "SDDLM loss")

# Gradient flows without NaN
model.train()
logits2 = model(xt, t_f)
l2, _ = sddlm_loss(logits2, x0, xt)
l2.backward()
for name, p in model.named_parameters():
    if p.grad is not None:
        no_nan_inf(p.grad, f"grad of {name}")
model.zero_grad()



section("TEST 4 · SDDLM-V1 loss  (Eq. 9)")

model.eval()
logits = model(xt, t_f).detach()

loss_v1, info_v1 = sddlm_v1_loss(logits, x0, xt, vocab_size=V)
no_nan_inf(loss_v1, "SDDLM-V1 loss")
check(math.isfinite(loss_v1.item()), f"V1 loss is finite  (got {loss_v1.item():.4f})")

# pos_nll must be positive (it IS a negative log probability, always > 0)
check(
    info_v1["pos_nll"] > 0.0,
    f"positive term (pos_nll) > 0  (got {info_v1['pos_nll']:.4f})",
)

# neg_term is E[log p(x̂)] which is always NEGATIVE (log of prob < 1)
# The loss adds it (not subtracts), so this term REDUCES the total loss
check(
    info_v1["neg_term"] < 0.0,
    f"negative term E[log p(x̂)] < 0  (got {info_v1['neg_term']:.4f})  "
    f"← correct: log-prob of random token is always negative",
)

# Stability test: extreme logits should NOT produce NaN (epsilon protects this)
extreme_logits = torch.full((B, L, V), -1e9)
extreme_logits[:, :, 0] = 1e9  # model is 100% confident about token 0
loss_extreme, _ = sddlm_v1_loss(extreme_logits, x0, xt, vocab_size=V)
no_nan_inf(loss_extreme, "V1 loss at extreme logits")
check(
    loss_extreme.item() == 0.0 or math.isfinite(loss_extreme.item()),
    "extreme logits don't crash V1 loss",
)

fake_same = x0.clone()
loss_v1_zero, _ = sddlm_v1_loss(logits, x0, fake_same, vocab_size=V)
check(
    loss_v1_zero.item() == 0.0,
    f"V1 loss=0 when no tokens corrupted (got {loss_v1_zero.item():.6f})",
)

model.train()
logits3 = model(xt, t_f)
l3, _ = sddlm_v1_loss(logits3, x0, xt, vocab_size=V)
l3.backward()
for name, p in model.named_parameters():
    if p.grad is not None:
        no_nan_inf(p.grad, f"V1 grad of {name}")
model.zero_grad()

logits4 = model(xt, t_f)
l4, _ = compute_loss(logits4, x0, xt, cfg.loss)
check(
    math.isfinite(l4.item()),
    f"compute_loss dispatcher returns finite value (got {l4.item():.4f})",
)

section("TEST 5 · Training loop — loss decreases over 200 steps")

torch.manual_seed(0)
model2 = DiffusionLM(cfg.model)
model2.train()
opt = torch.optim.AdamW(model2.parameters(), lr=3e-4)


x0_fixed = torch.randint(0, V, (B, L))

losses = []
for step in range(200):
    t_idx = sched.sample_t(B, device)
    xt_s, _ = diffusion.q_sample(x0_fixed, t_idx)
    t_f_s = sched.t_to_float(t_idx)

    logits_s = model2(xt_s, t_f_s)
    loss_s, _ = compute_loss(logits_s, x0_fixed, xt_s, cfg.loss)

    opt.zero_grad()
    loss_s.backward()
    nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
    opt.step()

    losses.append(loss_s.item())

first_20 = sum(losses[:20]) / 20
last_20 = sum(losses[-20:]) / 20

abs_drop = first_20 - last_20
check(
    last_20 < first_20,
    f"loss decreases: {first_20:.4f} → {last_20:.4f}  (drop={abs_drop:.4f})",
)
check(
    abs_drop > 0.5, f"absolute loss drop > 0.5 nats over 200 steps (got {abs_drop:.4f})"
)

print(
    f"       loss trajectory: {losses[0]:.4f} → {losses[49]:.4f} → {losses[99]:.4f} → {losses[-1]:.4f}"
)


section("TEST 6 · Generation (reverse diffusion sampling)")

model_gen = DiffusionLM(cfg.model)
model_gen.eval()

x_gen = diffusion.sample(
    model=model_gen,
    batch_size=4,
    seq_len=L,
    num_steps=10,
    device=device,
)

check(x_gen.shape == (4, L), f"sample() output shape is (4, {L})", str(x_gen.shape))
check(x_gen.dtype == torch.long, "generated tokens are LongTensor")
check(
    bool((x_gen >= 0).all() and (x_gen < V).all()),
    f"all generated token ids are valid (0 ≤ id < {V})",
)

n_unique = x_gen.unique().numel()
check(
    n_unique > 5,
    f"fresh model generates >5 unique tokens (got {n_unique})"
    f"  — random weights → near-uniform → diverse",
)

print(f"       unique tokens in 4×{L} generated sequences: {n_unique} / {V}")


section("TEST 7 · Gradient norm sanity")

model2.train()
x0_gn = torch.randint(0, V, (B, L))
t_gn = sched.sample_t(B, device)
xt_gn, _ = diffusion.q_sample(x0_gn, t_gn)
t_fgn = sched.t_to_float(t_gn)

logits_gn = model2(xt_gn, t_fgn)
loss_gn, _ = compute_loss(logits_gn, x0_gn, xt_gn, cfg.loss)
loss_gn.backward()

raw_norm = nn.utils.clip_grad_norm_(model2.parameters(), float("inf"))
check(
    raw_norm.item() < 1e6, f"raw grad norm is not exploding (got {raw_norm.item():.2f})"
)

nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
post_norm = nn.utils.clip_grad_norm_(model2.parameters(), float("inf"))
check(
    post_norm.item() <= 1.0 + 1e-4,
    f"after clip, recomputed norm <= 1.0 (got {post_norm.item():.4f})",
)

has_nan = any(
    p.grad is not None and torch.isnan(p.grad).any() for p in model2.parameters()
)
check(not has_nan, "no NaN in gradients after backward")


section("TEST 8 · Time embedding edge cases")

from src.model import SinusoidalTimeEmbedding

te = SinusoidalTimeEmbedding(cfg.model.d_model)

# t=0 and t=1 (boundary values)
t_edges = torch.tensor([0.0, 0.5, 1.0])
emb = te(t_edges)
check(
    emb.shape == (3, cfg.model.d_model),
    f"time emb shape (3,{cfg.model.d_model})",
    str(emb.shape),
)
no_nan_inf(emb, "time embedding at t=0, 0.5, 1.0")

# Different t values must give different embeddings
diff01 = (emb[0] - emb[1]).norm().item()
check(diff01 > 1e-3, f"t=0 and t=0.5 give different embeddings (L2={diff01:.4f})")

diff12 = (emb[1] - emb[2]).norm().item()
check(diff12 > 1e-3, f"t=0.5 and t=1.0 give different embeddings (L2={diff12:.4f})")

print(f"\n{'═'*50}")
print(f"\033[92m  ALL TESTS PASSED — code is correct, safe to train.\033[0m")
print(f"{'═'*50}\n")
