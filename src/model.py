import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class SinusoidalTimeEmbedding(nn.Module):
    
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )  # (half,)
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.mlp(emb)

class AdaLN(nn.Module):
   
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
        
class RotaryEmbedding(nn.Module):

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
        
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)  
        def to_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Apply RoPE to Q and K only (not V)
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # diffusion is BIDIRECTIONAL — no causal mask
        ) 

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class SwiGLUFFN(nn.Module):
       def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

class DiTBlock(nn.Module):

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



class DiffusionLM(nn.Module):
    
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

        self.out_head.weight = self.token_emb.weight

       
        self.apply(self._init_weights)
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
        
        x = self.token_emb(xt)  # (B, L, d_model)
        te = self.time_emb(t)  # (B, d_model)
        for block in self.blocks:
            x = block(x, te)
        return self.out_head(self.out_norm(x))  # (B, L, V)
