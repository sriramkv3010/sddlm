"""
config.py — all hyperparameters in one place.

Why dataclasses and not YAML?
  - No extra dependency
  - Type-checked at runtime
  - IDE autocomplete works
  - Can be imported and modified programmatically in experiments
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Vocabulary from GPT-2 tokenizer
    vocab_size: int = 50257

    # Transformer width — 256 is a "nano" model, trainable on CPU/MPS
    # Scale to 512 or 768 on a GPU cluster
    d_model: int = 256

    # Number of attention heads — must divide d_model evenly
    n_heads: int = 4

    # Depth of the transformer
    n_layers: int = 6

    # Feed-forward inner dimension (SwiGLU uses this as the "up-projection")
    d_ff: int = 1024

    # Dropout applied to attention weights and FFN activations
    dropout: float = 0.1

    # Maximum sequence length the model is built for
    # RoPE cache is pre-built up to this length
    max_seq_len: int = 128


@dataclass
class DiffusionConfig:
    # Discrete number of timesteps T
    # At training: t ~ Uniform[0, T-1]; at inference: we can use fewer steps
    num_timesteps: int = 1000

    # Noise schedule type: "cosine" (smoother) or "linear"
    # Cosine is better for text — avoids near-zero alpha_t until very late
    schedule: str = "cosine"

    # Minimum alpha_t value to avoid exact zero (numerical stability)
    eps: float = 1e-4


@dataclass
class TrainingConfig:
    # Where to find train.txt and test.txt
    data_dir: str = "data/wikitext2"

    # Tokens processed per gradient step = batch_size * seq_len
    # 16 * 128 = 2048 tokens/step — fine for 16GB RAM on M4
    batch_size: int = 16

    # AdamW learning rate — matches the paper (3e-4 for small models)
    learning_rate: float = 3e-4

    # Linear warmup: ramp LR from 0 to learning_rate over this many steps
    warmup_steps: int = 1000

    # Total training steps
    # WikiText-2 has ~2M tokens; 50k steps * 2048 tokens = 100M tokens seen
    max_steps: int = 50000

    # How often to print validation loss
    eval_every: int = 500

    # How often to save checkpoint
    save_every: int = 2000

    # Where to write checkpoints
    checkpoint_dir: str = "checkpoints"

    # "auto" → picks MPS on Apple Silicon, CUDA if available, else CPU
    device: str = "auto"

    # AdamW weight decay (paper uses 0 for small models)
    weight_decay: float = 0.0

    # Gradient clipping — prevents exploding gradients (important for diffusion)
    grad_clip: float = 1.0

    # AdamW beta2 (paper: 0.999 for small, 0.95 for large)
    adam_beta2: float = 0.999


@dataclass
class LossConfig:
    # "sddlm"    → Eq. (7): plain denoising on corrupted positions only
    # "sddlm_v1" → Eq. (9): + anti-uniform regularisation (negative gradient)
    loss_type: str = "sddlm_v1"

    # Added to probabilities BEFORE taking log — prevents log(0)
    # Paper says: "add small constant ε to pθ(x0|xt) and pθ(x̂|xt)"
    epsilon: float = 1e-6

    # Monte Carlo samples for the negative gradient term
    # 1 is what the paper uses; more = lower variance but slower
    n_neg_samples: int = 1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
