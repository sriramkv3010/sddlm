import math
import torch
import torch.nn.functional as F

class NoiseSchedule:
    
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
        return self.alphas[t_idx.cpu()].to(t_idx.device)

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        
        return torch.randint(0, self.T, (batch_size,), device=device)

    def t_to_float(self, t_idx: torch.Tensor) -> torch.Tensor:
        
        return t_idx.float() / self.T

class UniformDiffusion:

    def __init__(self, schedule: NoiseSchedule, vocab_size: int):
        self.schedule = schedule
        self.vocab_size = V = vocab_size

    def q_sample(
        self,
        x0: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      
        B, L = xt.shape
        device = xt.device

    
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
