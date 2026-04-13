"""
evaluate.py — Generative Perplexity (Gen PPL) and Entropy.

Usage:
    cd sddlm/

    # Quick 3k checkpoint (fast check, numbers won't be great)
    python3 src/evaluate.py --checkpoint checkpoints/quick/quick_3k.pt --n_gen 50 --steps 64

    # Full trained checkpoint
    python3 src/evaluate.py --checkpoint checkpoints/final.pt --n_gen 200 --steps 128

    # Skip Gen PPL if no internet
    python3 src/evaluate.py --checkpoint checkpoints/final.pt --skip_ppl
"""

import os
import sys
import math
import argparse
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import DiffusionLM
from src.diffusion import NoiseSchedule, UniformDiffusion
from src.dataset import get_tokenizer
from src.sample import load_model


def generate_sequences(model, diffusion, n, seq_len, num_steps, device, batch_size=16):
    """Generate n sequences with progress bar."""
    all_ids = []
    model.eval()
    for start in range(0, n, batch_size):
        this_batch = min(batch_size, n - start)
        ids = diffusion.sample(
            model=model,
            batch_size=this_batch,
            seq_len=seq_len,
            num_steps=num_steps,
            device=device,
        )
        all_ids.extend(ids.tolist())
        done = len(all_ids)
        bar = "X" * (done * 30 // n) + "." * (30 - done * 30 // n)
        print(f"  Generating  [{bar}]  {done}/{n}", end="\r")
    print()
    return all_ids


def compute_entropy(generated_ids):
    """
    Shannon entropy over empirical token distribution (nats).

    H = -sum_v  p(v) * log(p(v))

    Interpretation:
      ~0 nats   = degenerate (one token repeated)
      ~5.3 nats = paper target on OWT
      ~10.8 nats = pure random (max for vocab_size=50257)
    """
    counter = Counter()
    total = 0
    for seq in generated_ids:
        counter.update(seq)
        total += len(seq)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def compute_gen_ppl(generated_ids, batch_size=8):
    """
    Score generated sequences under frozen GPT-2.

    Gen PPL = exp( -1/N * sum log p_GPT2(token | context) )

    Always runs on CPU to avoid MPS issues with HuggingFace.

    Interpretation:
      ~45  = paper SDDLM-V1 on OWT  (excellent)
      ~80  = paper Duo baseline      (good)
      ~300 = learning but not converged
      ~3000+ = barely trained
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\n  Loading GPT-2 scorer (first run downloads ~500 MB, cached after)...")
    eval_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cpu")
    eval_model.eval()
    print("  GPT-2 loaded.\n")

    total_nll = 0.0
    total_tokens = 0
    n_batches = math.ceil(len(generated_ids) / batch_size)

    with torch.no_grad():
        for i, start in enumerate(range(0, len(generated_ids), batch_size)):
            batch_seqs = generated_ids[start : start + batch_size]
            max_len = max(len(s) for s in batch_seqs)
            padded = [
                s + [eval_tok.eos_token_id] * (max_len - len(s)) for s in batch_seqs
            ]
            attn_masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in batch_seqs]

            input_ids = torch.tensor(padded, dtype=torch.long)
            attn_mask = torch.tensor(attn_masks, dtype=torch.long)
            labels = input_ids.clone()
            labels[attn_mask == 0] = -100

            outputs = eval_model(
                input_ids=input_ids, attention_mask=attn_mask, labels=labels
            )
            n_real = int((attn_mask == 1).sum().item())
            total_nll += outputs.loss.item() * n_real
            total_tokens += n_real
            print(f"  Scoring [{i+1}/{n_batches} batches]", end="\r")

    print()
    del eval_model
    return math.exp(total_nll / max(total_tokens, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt")
    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument(
        "--n_show", type=int, default=4, help="Number of sample texts to print"
    )
    args = parser.parse_args()

    # device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  SDDLM Evaluation")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(f"  n_gen      : {args.n_gen} sequences x {args.seq_len} tokens")
    print(f"  steps      : {args.steps} denoising steps")
    print(f"{'='*60}\n")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at '{args.checkpoint}'")
        print("Run  python3 src/train.py  or  python3 quick_train.py  first.")
        sys.exit(1)

    # load
    model, cfg = load_model(args.checkpoint, device)
    schedule = NoiseSchedule(
        num_timesteps=cfg.diffusion.num_timesteps,
        schedule=cfg.diffusion.schedule,
        eps=cfg.diffusion.eps,
    )
    diffusion = UniformDiffusion(schedule, vocab_size=cfg.model.vocab_size)
    tokenizer = get_tokenizer()

    # generate
    print(f"Step 1/3 -- Generating {args.n_gen} sequences...\n")
    all_ids = generate_sequences(
        model,
        diffusion,
        n=args.n_gen,
        seq_len=args.seq_len,
        num_steps=args.steps,
        device=device,
    )
    print(f"  Done -- {len(all_ids)*args.seq_len:,} tokens generated\n")

    # show sample text
    print(f"{'─'*60}")
    print(f"  Sample generated text ({args.n_show} of {args.n_gen})")
    print(f"{'─'*60}")
    for i in range(min(args.n_show, len(all_ids))):
        text = tokenizer.decode(all_ids[i], skip_special_tokens=True)
        # truncate to ~280 chars
        display = text[:280].replace("\n", " ")
        print(f"\n  [{i+1}] {display}")
    print(f"\n{'─'*60}\n")

    # entropy
    print("Step 2/3 -- Computing Entropy...")
    entropy = compute_entropy(all_ids)
    max_h = math.log(cfg.model.vocab_size)
    print(f"\n  Entropy : {entropy:.4f} nats")
    print(f"            max possible = {max_h:.2f} nats (pure random)")
    print(f"            paper SDDLM-V1 on OWT = ~5.31 nats\n")

    # gen ppl
    if not args.skip_ppl:
        print("Step 3/3 -- Computing Gen PPL (GPT-2 scorer)...")
        gen_ppl = compute_gen_ppl(all_ids)
        print(f"\n  Gen PPL : {gen_ppl:.2f}")
        print(f"            lower = better")
        print(f"            paper SDDLM-V1 on OWT = ~45.18\n")
        print(f"            paper Duo baseline    = ~80.43\n")
    else:
        gen_ppl = None
        print("Step 3/3 -- Gen PPL skipped (--skip_ppl)\n")

    # summary
    print(f"{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Entropy  : {entropy:.4f} nats   [paper target ~5.31]")
    if gen_ppl is not None:
        print(f"  Gen PPL  : {gen_ppl:.2f}         [paper target ~45.18]")
    else:
        print(f"  Gen PPL  : skipped")
    print(f"{'='*60}")
    print(f"\n  Note: this model trained on WikiText-2 (2M tokens).")
    print(f"  Paper trained on OpenWebText (much larger dataset).")
    print(f"  Expect higher Gen PPL than paper — that is normal.\n")


if __name__ == "__main__":
    main()
