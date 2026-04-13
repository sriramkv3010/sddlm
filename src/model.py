"""
model.py — Diffusion Transformer (DiT-style) language model.

Architecture chain:
    token ids (B, L)
        ↓  nn.Embedding
    token vectors (B, L, d_model)
        ↓  + time conditioning via AdaLN in every block
    N × DiTBlock
        ↓  LayerNorm → Linear
    logits (B, L, V)   ← p_θ(x_0 | x_t)

Key design choices (all matching the paper):
  • RoPE positional encoding instead of learned absolute positions
      → relative position information, generalises to longer sequences
  • AdaLN time conditioning (Adaptive Layer Norm)
      → time signal modulates the SCALE of every layer, not just prepended
  • SwiGLU feed-forward
      → smoother activations, slightly better empirically than ReLU/GELU
  • Weight tying between token embedding and output head
      → saves ~12M parameters, improves generalisation at this scale
  • No causal mask — diffusion models are bidirectional
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Time embedding
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar timestep t ∈ [0, 1] to a d_model-dimensional vector.

    Two-stage process:
      1. Sinusoidal encoding at d_model/2 different frequencies
         (borrowed from the positional encoding of the original Transformer)
      2. Two-layer MLP with SiLU activation to allow non-linear mixing

    Why sinusoidal first?
      • Smooth interpolation between nearby timesteps
      • Each frequency captures a different "scale" of the noise level
      • No learnable parameters in the frequency part → stable initialisation

    Why MLP on top?
      • Allows the model to learn non-linear relationships between frequency
        components and the actual noise semantics it needs to express
    """

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model

        # MLP: project sinusoidal features → richer time representation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B,) float, values in [0, 1]
        Returns:
            emb : (B, d_model)
        """
        half = self.d_model // 2
        # log-spaced frequencies: ω_i = exp(-i * log(10000) / (half-1))
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )  # (half,)
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Layer Norm (AdaLN)
# ─────────────────────────────────────────────────────────────────────────────


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization conditioned on a time embedding.

    Standard LayerNorm normalises activations then multiplies by a
    learnable γ and adds a learnable β — both FIXED scalars.

    AdaLN makes γ and β FUNCTIONS of the timestep:
        AdaLN(x, t) = LayerNorm(x) * (1 + γ(t)) + β(t)

    The (1 + ...) ensures that at initialisation (weights = 0) the
    behaviour is identical to plain LayerNorm, so training starts stably.

    Why AdaLN over simple concatenation or FiLM?
      • Operates multiplicatively on the normalised signal
        → can amplify or silence whole feature groups depending on noise level
      • One linear layer per block, very low overhead
      • Proven in DiT, ControlNet, and many modern diffusion backbones
    """

    def __init__(self, d_model: int):
        super().__init__()
        # elementwise_affine=False: we supply our own γ and β from time emb
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Predict 2 * d_model: first half = scale (γ), second half = shift (β)
        self.modulation = nn.Linear(d_model, 2 * d_model, bias=True)
        # Zero-init → identity at the start of training
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : (B, L, d_model)
            time_emb : (B, d_model)
        Returns:
            out      : (B, L, d_model)
        """
        scale, shift = self.modulation(time_emb).chunk(2, dim=-1)
        # Broadcast over sequence length L
        scale = scale[:, None, :]  # (B, 1, d_model)
        shift = shift[:, None, :]  # (B, 1, d_model)
        return self.norm(x) * (1.0 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embedding (RoPE)
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """
    RoPE: applies a position-dependent rotation to query and key vectors.

    Unlike absolute positional embeddings (added to token embeddings once),
    RoPE is applied INSIDE attention to Q and K so that the dot product
    Q_m · K_n naturally captures the RELATIVE position (m - n).

    Benefits:
      • Relative position information without explicit relative bias matrices
      • Generalises to sequence lengths longer than seen at training time
      • No learnable parameters — all geometric, derived from position index
    """

    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        # θ_i = 10000^(-2i/d_head) for i = 0, 1, ..., d_head/2 - 1
        theta = 1.0 / (
            10000 ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head)
        )
        self.register_buffer("theta", theta)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len, dtype=self.theta.dtype)
        freqs = torch.outer(pos, self.theta)  # (seq_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_head)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """(x1, x2, ..., xn/2, xn/2+1, ..., xn) → (-xn/2+1, ..., -xn, x1, ..., xn/2)"""
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, n_heads, L, d_head)
        Returns:
            rotated x of same shape
        """
        L = x.shape[2]
        cos = self.cos_cache[:L][None, None]  # (1, 1, L, d_head)
        sin = self.sin_cache[:L][None, None]
        return x * cos + self._rotate_half(x) * sin


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Single matrix for Q, K, V — one matmul instead of three
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L, d_model)
        Returns:
            out : (B, L, d_model)
        """
        B, L, D = x.shape
        # Project to Q, K, V
        q, k, v = self.qkv(x).split(D, dim=-1)  # each (B, L, d_model)

        # Reshape to multi-head: (B, H, L, d_head)
        def to_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Apply RoPE to Q and K only (not V)
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        # PyTorch 2.0+ fused implementation (faster + less memory on MPS/CUDA)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # diffusion is BIDIRECTIONAL — no causal mask
        )  # (B, H, L, d_head)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward
# ─────────────────────────────────────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward: FFN(x) = (SiLU(gate(x)) ⊙ up(x)) · down

    Why SwiGLU over standard GELU FFN?
      • Gating mechanism selectively suppresses irrelevant dimensions
      • Empirically 0.5-1 ppl point better at the same parameter count
      • Used in LLaMA, PaLM, and the large-model version of the paper
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─────────────────────────────────────────────────────────────────────────────
# DiT Block
# ─────────────────────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """
    One transformer block conditioned on the timestep via AdaLN.

    Pre-norm layout (applies norm BEFORE attention/FFN, not after):
        x ← x + Attention(AdaLN₁(x, t))
        x ← x + FFN(AdaLN₂(x, t))

    Why pre-norm?
      • Gradients flow more stably through the residual stream
      • Standard in modern LLMs (GPT-2+, LLaMA, etc.)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.norm_attn = AdaLN(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.norm_ffn = AdaLN(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x, time_emb))
        x = x + self.ffn(self.norm_ffn(x, time_emb))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────


class DiffusionLM(nn.Module):
    """
    p_θ(x_0 | x_t, t)  —  the denoising network.

    Input : noisy token sequence x_t  (B, L) LongTensor
             timestep fraction t       (B,)  FloatTensor in [0, 1]
    Output: logits over the vocabulary (B, L, V) — un-normalised log probs

    The logits represent the model's best guess for the clean token at
    every position, given the corrupted sequence and the noise level t.

    Weight tying:
      output_head.weight  ←  token_emb.weight
    This reduces ~12M parameters at vocab_size=50257, d_model=256
    and acts as a regulariser (the output space has the same geometry
    as the embedding space).
    """

    def __init__(self, cfg):
        super().__init__()
        d, V = cfg.d_model, cfg.vocab_size

        self.token_emb = nn.Embedding(V, d)
        self.time_emb = SinusoidalTimeEmbedding(d)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.max_seq_len)
                for _ in range(cfg.n_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, V, bias=False)

        # Weight tying
        self.out_head.weight = self.token_emb.weight

        # Initialise weights — GPT-style scaled init
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 * n_layers) for depth stability
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or "down_proj.weight" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

        n = sum(p.numel() for p in self.parameters())
        print(f"DiffusionLM  |  parameters: {n:,}  ({n/1e6:.2f} M)")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt : (B, L)  — noisy token ids
            t  : (B,)    — timestep fractions in [0, 1]
        Returns:
            logits : (B, L, V)
        """
        x = self.token_emb(xt)  # (B, L, d_model)
        te = self.time_emb(t)  # (B, d_model)
        for block in self.blocks:
            x = block(x, te)
        return self.out_head(self.out_norm(x))  # (B, L, V)
