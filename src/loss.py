"""
loss.py — SDDLM and SDDLM-V1 training objectives.

─────────────────────────────────────────────────────────
SDDLM  (Eq. 7)
─────────────────────────────────────────────────────────
L_SDDLM = E_{x_0, t, x_t} [
    Σ_l  -log p_θ(x_0^l | x_t)  ·  1(x_0^l ≠ x_t^l)
]

Meaning:
  • Standard cross-entropy loss on the clean token at each position
  • BUT only applied where the token was actually corrupted (x_t ≠ x_0)
  • Positions with x_t = x_0 are EXCLUDED from the gradient

Why exclude unchanged positions?
  • Corrupted positions: model must denoise  (useful signal)
  • Unchanged positions: model reconstructs identity  (trivial, hurts training)
  • Mixing the two objectives causes gradient conflict → training collapse

─────────────────────────────────────────────────────────
SDDLM-V1  (Eq. 9)
─────────────────────────────────────────────────────────
L_V1 = E [
    Σ_l  ( -log p_θ(x_0^l | x_t) + E_{x̂~U(V)}[log p_θ(x̂^l | x_t)] )
          ·  1(x_0^l ≠ x_t^l)
]

Derivation from KL divergence (Eq. 8 → 9):
  The regularisation term is  -KL(U ∥ p_θ(·|x_t))  where U = 1/V.

  KL(U ∥ p) = Σ_v U(v) log[U(v) / p(v)]
             = -E_{x̂~U}[log p(x̂)] - log|V|

  Subtracting KL from the loss (i.e. minimising -KL → maximising KL):
    L = L_SDDLM  -  KL(U ∥ p_θ)   (ignoring the constant log|V|)
      = -log p_θ(x_0)  +  E_{x̂~U}[log p_θ(x̂)]

Gradient decomposition (Eq. 10):
  ∇_θ L = -∇_θ log p_θ(x_0)        ← ATTRACTIVE  (towards correct token)
          + E[∇_θ log p_θ(x̂)]       ← REPULSIVE   (away from random tokens)

  Update rule  θ ← θ - η ∇_θ L:
    • +η ∇_θ log p_θ(x_0)  → INCREASES p_θ(x_0) ✓
    • -η ∇_θ log p_θ(x̂)   → DECREASES p_θ(x̂)  ✓

  Net effect: probability concentrates on the correct token,
  distribution becomes sharper → better generation quality.

Why KL(U ∥ p_θ) and NOT KL(p_θ ∥ U)?
  • KL(p_θ ∥ U) has gradient ∝ (p_θ - U) / U ≈ (p_θ - 1/V) / (1/V) = V·p_θ - 1
    This is dominated by high-probability tokens → mode-seeking behaviour.
  • KL(U ∥ p_θ) has gradient ∝ 1/p_θ
    Penalises LOW probability for any token — forces a SPREAD distribution.
  • We SUBTRACT KL(U∥p_θ) → we MAXIMISE it → the distribution becomes
    as UN-uniform (peaked) as possible. This fights the uniform-prior
    corruption that tends to flatten the output distribution.

Numerical stability:
  Paper says: "add ε to p_θ(x_0|x_t) and p_θ(x̂|x_t) before taking log".
  We apply ε to the probabilities (after softmax), not to the logits.
  This prevents log(0) when a token has near-zero probability.
"""

import torch
import torch.nn.functional as F


def sddlm_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    SDDLM base loss (Eq. 7).

    Args:
        logits  : (B, L, V)  un-normalised model outputs
        x0      : (B, L)     clean token ids
        xt      : (B, L)     corrupted token ids
        epsilon : float      stability constant added to probs before log

    Returns:
        loss    : scalar
        info    : dict with diagnostic values (fraction corrupted, etc.)
    """
    # ── numerical-stable log-probs ─────────────────────────────────────────
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    log_probs = torch.log(probs + epsilon)  # (B, L, V)

    # ── negative log-likelihood at correct token ───────────────────────────
    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # ── corrupted-position mask ────────────────────────────────────────────
    # 1 where token was changed by the forward process, 0 otherwise
    mask = (x0 != xt).float()  # (B, L)

    n_corrupted = mask.sum()

    if n_corrupted == 0:
        # Edge case: t ≈ 0, nothing was corrupted — zero loss
        dummy = (logits * 0).sum()  # keeps graph alive
        return dummy, {"frac_corrupted": 0.0, "n_corrupted": 0}

    loss = (nll * mask).sum() / n_corrupted

    info = {
        "frac_corrupted": (n_corrupted / mask.numel()).item(),
        "n_corrupted": int(n_corrupted.item()),
    }
    return loss, info


def sddlm_v1_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    vocab_size: int,
    epsilon: float = 1e-6,
    n_neg_samples: int = 1,
) -> tuple[torch.Tensor, dict]:
    """
    SDDLM-V1 loss (Eq. 9): SDDLM + anti-uniform regularisation.

    The negative gradient term is estimated by sampling n_neg_samples
    random tokens x̂ ~ U(V) per position and averaging their log-probs.
    n_neg_samples=1 (paper default) is sufficient because we average over
    many gradient steps.

    Args:
        logits        : (B, L, V)
        x0            : (B, L)  clean token ids
        xt            : (B, L)  corrupted token ids
        vocab_size    : int     |V|
        epsilon       : float   stability constant
        n_neg_samples : int     Monte Carlo samples for E[log p(x̂)]

    Returns:
        loss : scalar
        info : dict
    """
    B, L, V = logits.shape

    # ── numerical-stable probabilities ────────────────────────────────────
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    log_probs = torch.log(probs + epsilon)  # (B, L, V)

    # ── POSITIVE term: -log p_θ(x_0 | x_t) ──────────────────────────────
    nll_pos = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # ── NEGATIVE term: +E_{x̂~U}[log p_θ(x̂ | x_t)] ───────────────────────
    # Monte Carlo estimate: average over n_neg_samples random tokens
    neg_terms = []
    for _ in range(n_neg_samples):
        # Sample one random token per position
        x_hat = torch.randint(0, vocab_size, (B, L), device=logits.device)
        log_p_hat = log_probs.gather(-1, x_hat.unsqueeze(-1)).squeeze(-1)  # (B,L)
        neg_terms.append(log_p_hat)

    # Average over samples: E_{x̂~U}[log p(x̂)]
    neg_term = torch.stack(neg_terms, dim=0).mean(dim=0)  # (B, L)

    # ── Combined SDDLM-V1 objective ────────────────────────────────────────
    # loss_per_pos = -log p(x_0) + E[log p(x̂)]
    # Minimising this:
    #   • decreases -log p(x_0) → p(x_0) ↑   [attractive gradient]
    #   • decreases  E[log p(x̂)] → p(x̂) ↓   [repulsive gradient]
    loss_per_pos = nll_pos + neg_term  # (B, L)

    # ── Corrupted-position mask ────────────────────────────────────────────
    mask = (x0 != xt).float()  # (B, L)
    n_corrupted = mask.sum()

    if n_corrupted == 0:
        dummy = (logits * 0).sum()
        return dummy, {
            "frac_corrupted": 0.0,
            "n_corrupted": 0,
            "pos_nll": 0.0,
            "neg_term": 0.0,
        }

    loss = (loss_per_pos * mask).sum() / n_corrupted

    info = {
        "frac_corrupted": (n_corrupted / mask.numel()).item(),
        "n_corrupted": int(n_corrupted.item()),
        "pos_nll": ((nll_pos * mask).sum() / n_corrupted).item(),
        "neg_term": ((neg_term * mask).sum() / n_corrupted).item(),
    }
    return loss, info


def compute_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    loss_cfg,
) -> tuple[torch.Tensor, dict]:
    """
    Dispatcher — calls the right loss function based on config.

    Args:
        logits   : (B, L, V)
        x0       : (B, L)  clean tokens
        xt       : (B, L)  corrupted tokens
        loss_cfg : LossConfig dataclass
    Returns:
        loss, info_dict
    """
    V = logits.shape[-1]

    if loss_cfg.loss_type == "sddlm":
        return sddlm_loss(logits, x0, xt, epsilon=loss_cfg.epsilon)

    elif loss_cfg.loss_type == "sddlm_v1":
        return sddlm_v1_loss(
            logits,
            x0,
            xt,
            vocab_size=V,
            epsilon=loss_cfg.epsilon,
            n_neg_samples=loss_cfg.n_neg_samples,
        )

    else:
        raise ValueError(f"Unknown loss type: '{loss_cfg.loss_type}'")
