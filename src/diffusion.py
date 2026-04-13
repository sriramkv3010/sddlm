"""
diffusion.py — Uniform-State Diffusion forward and reverse processes.

Forward process  q(x_t | x_0):
  Each token independently:
    with prob α_t  → keep original token
    with prob 1-α_t → replace with a uniformly random token from V

  In distribution notation (Eq. 3 of paper):
    q_t(· | x_0^l ; α_t) = Cat(· ; α_t · x_0^l + (1-α_t) · π)
  where π = 1/V (uniform over vocabulary).

Reverse process  p_θ(x_0 | x_t):
  The model predicts the clean token at every position.
  We sample from this prediction and re-corrupt to level s < t
  (ancestral sampling / DDIM-style for discrete space).

Why uniform prior instead of mask?
  • Analogy to Gaussian diffusion: x_T ~ N(0,I) is structureless noise.
    The uniform distribution is the discrete equivalent of white noise.
  • Enables consistency distillation and other continuous-space speedups
    (deterministic mappings from noise → data exist, unlike MDMs).
  • MDMs can only unmask; USDMs can "re-token", giving more flexibility.
"""

import math
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────


class NoiseSchedule:
    """
    Pre-computes α_t for every discrete timestep t ∈ {0, …, T-1}.

    Cosine schedule (default):
        α_t = cos²(π·t/(2·T))

    Properties:
      • α_0  ≈ 1.0  (no corruption — x_t ≈ x_0)
      • α_T  ≈ 0.0  (full corruption — x_t ~ Uniform(V))
      • Smoothly decreasing — avoids abrupt noise injection
        that causes training instability
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        eps: float = 1e-4,
    ):
        self.T = num_timesteps
        self.eps = eps

        # t / T gives fractions in (0, 1]
        t_frac = torch.arange(num_timesteps, dtype=torch.float) / num_timesteps

        if schedule == "cosine":
            alphas = torch.cos(math.pi * t_frac / 2).pow(2)
        elif schedule == "linear":
            alphas = 1.0 - t_frac
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        # Clamp to [eps, 1] so log(α) is never -inf
        self.alphas = alphas.clamp(eps, 1.0)  # (T,)

    def get_alpha(self, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_idx : (B,) LongTensor with values in [0, T-1]
        Returns:
            α     : (B,) FloatTensor
        """
        return self.alphas[t_idx.cpu()].to(t_idx.device)

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniform random timesteps for a training batch."""
        return torch.randint(0, self.T, (batch_size,), device=device)

    def t_to_float(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Convert integer index → fraction in [0, 1) for the model."""
        return t_idx.float() / self.T


# ─────────────────────────────────────────────────────────────────────────────
# Uniform-State Diffusion
# ─────────────────────────────────────────────────────────────────────────────


class UniformDiffusion:
    """
    Wraps the forward (corruption) and reverse (generation) processes.
    """

    def __init__(self, schedule: NoiseSchedule, vocab_size: int):
        self.schedule = schedule
        self.vocab_size = V = vocab_size

    # ── forward process ───────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0 ; α_t).

        Implementation:
            For each token position independently:
                keep  = Bernoulli(α_t)
                x_t   = x_0        if keep
                      = Uniform(V)  otherwise

        This is exact because:
          P(x_t = x_0^l) = α_t + (1-α_t)/V  ≈ α_t  for large V
          P(x_t = v≠x_0) = (1-α_t)/V

        The mask 1(x_t ≠ x_0) used in the loss is NOT identical to the
        keep indicator, because a uniform sample could accidentally equal x_0
        with probability (1-α_t)/V.  For V=50257 this is negligible
        (~2e-5 per position), so in practice the mask is an excellent proxy.

        Args:
            x0    : (B, L)  clean token ids
            t_idx : (B,)    timestep indices
        Returns:
            xt    : (B, L)  corrupted token ids
            alpha : (B,)    noise levels (useful for debugging / logging)
        """
        B, L = x0.shape

        alpha = self.schedule.get_alpha(t_idx)  # (B,)
        alpha_bl = alpha[:, None].expand(B, L)  # (B, L)

        # Keep mask: True → keep x_0, False → replace with noise
        keep = torch.bernoulli(alpha_bl).bool()  # (B, L)

        # Uniform noise tokens — independent of x_0
        noise = torch.randint(0, self.vocab_size, (B, L), device=x0.device)

        xt = torch.where(keep, x0, noise)
        return xt, alpha

    # ── reverse process (sampling / generation) ────────────────────────────

    @torch.no_grad()
    def p_sample_step(
        self,
        model,
        xt: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        One denoising step: x_t → x_s  (s < t).

        Strategy — "predict x_0, then re-corrupt to s":
          1. Run model → p_θ(x_0 | x_t)
          2. Sample x_0_hat from this distribution  (or take argmax)
          3. Re-sample from q(x_s | x_0_hat) using the forward process

        This is analogous to DDIM for continuous diffusion:
          it gives a principled way to skip steps while staying on the
          data manifold defined by the model's learned distribution.

        Alternative: Eq. (5) ancestral sampling — more accurate but slower
        and more complex to implement correctly.

        Args:
            model : DiffusionLM
            xt    : (B, L)  current noisy sequence
            t_idx : (B,)    current timestep
            s_idx : (B,)    target timestep (s < t)
        Returns:
            xs    : (B, L)  less noisy sequence at time s
        """
        B, L = xt.shape
        device = xt.device

        # Step 1: model prediction
        t_float = self.schedule.t_to_float(t_idx)  # (B,) in [0,1)
        logits = model(xt, t_float)  # (B, L, V)
        probs = F.softmax(logits, dim=-1)  # (B, L, V)

        # Step 2: sample x0_hat from p_θ(x_0 | x_t)
        # Move to CPU for multinomial (avoids rare MPS/CUDA multinomial issues)
        probs_cpu = probs.view(B * L, self.vocab_size).cpu()
        x0_hat = torch.multinomial(probs_cpu, num_samples=1).squeeze(-1)
        x0_hat = x0_hat.view(B, L).to(device)  # (B, L)

        # Step 3: re-corrupt x0_hat to noise level s
        # xs ~ q(x_s | x_0_hat ; α_s)
        xs, _ = self.q_sample(x0_hat, s_idx)
        return xs

    @torch.no_grad()
    def sample(
        self,
        model,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Full generation: start from uniform noise, iteratively denoise.

        Args:
            model      : trained DiffusionLM
            batch_size : number of sequences to generate
            seq_len    : length of each sequence (tokens)
            num_steps  : number of denoising steps (≤ T, can be much smaller)
            device     : where to run
        Returns:
            x0 : (B, L)  generated token ids
        """
        model.eval()
        T = self.schedule.T

        # Start from pure uniform noise: x_T ~ Uniform(V)
        xt = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)

        # Build evenly-spaced denoising trajectory: T-1, ..., 0
        # Example with num_steps=10, T=1000: [999, 899, 799, ..., 99, 0]
        step = max(1, T // num_steps)
        t_seq = list(range(T - 1, 0, -step))
        if t_seq[-1] != 0:
            t_seq.append(0)

        # Iteratively denoise
        for i in range(len(t_seq) - 1):
            t = t_seq[i]
            s = t_seq[i + 1]
            t_idx = torch.full((batch_size,), t, dtype=torch.long, device=device)
            s_idx = torch.full((batch_size,), s, dtype=torch.long, device=device)
            xt = self.p_sample_step(model, xt, t_idx, s_idx)

        # Final step at t=0: take argmax of model prediction
        # (at t=0, α ≈ 1, model should be very confident)
        t_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        t_float = self.schedule.t_to_float(t_idx)
        logits = model(xt, t_float)
        x0 = logits.argmax(dim=-1)  # (B, L)

        return x0
