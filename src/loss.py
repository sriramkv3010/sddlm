import torch
import torch.nn.functional as F


def sddlm_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    log_probs = torch.log(probs + epsilon)  # (B, L, V)

    # ── negative log likelihood at correct token 
    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
    mask = (x0 != xt).float()  # (B, L)

    n_corrupted = mask.sum()

    if n_corrupted == 0:
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
    B, L, V = logits.shape
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    log_probs = torch.log(probs + epsilon)  # (B, L, V)

    nll_pos = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

    neg_terms = []
    for _ in range(n_neg_samples):
        # Sample one random token per position
        x_hat = torch.randint(0, vocab_size, (B, L), device=logits.device)
        log_p_hat = log_probs.gather(-1, x_hat.unsqueeze(-1)).squeeze(-1)  # (B,L)
        neg_terms.append(log_p_hat)

    # Average over samples: E_{x̂~U}[log p(x̂)]
    neg_term = torch.stack(neg_terms, dim=0).mean(dim=0)  # (B, L)
    loss_per_pos = nll_pos + neg_term  # (B, L)
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
