from dataclasses import dataclass, field
@dataclass
class ModelConfig:
   
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128

@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    schedule: str = "cosine"
    eps: float = 1e-4

@dataclass
class TrainingConfig:
    data_dir: str = "data/wikitext2"
    batch_size: int = 16
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 50000
    eval_every: int = 500
    save_every: int = 2000
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    adam_beta2: float = 0.999

@dataclass
class LossConfig:
    loss_type: str = "sddlm_v1"
    epsilon: float = 1e-6
    n_neg_samples: int = 1

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
