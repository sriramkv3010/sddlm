# Simple Denoising Diffusion Language Model (SDDLM)

Implementation of **Simple Denoising Diffusion Language Models**, a diffusion-based approach for discrete sequence generation that leverages simplified denoising objectives for efficient and stable training.

---

## Overview

This project presents a **diffusion-based language modeling framework** that generates text by progressively refining noisy token sequences. Unlike autoregressive models, SDDLM models the entire sequence jointly and performs generation through iterative denoising.

Key characteristics:

* Initialization from a **uniform token distribution**
* Iterative **denoising-based generation**
* Parallel token refinement across the sequence
* Efficient training via **selective denoising objectives**

---

## Methodology

The implementation follows the **SDDLM-V1 formulation**, which introduces:

* **Selective Denoising Objective**
  Training focuses only on corrupted tokens, improving stability and efficiency.

* **Denoising-Based Learning Framework**
  The model learns to reconstruct clean sequences from progressively noised inputs.

* **Regularized Training Dynamics**
  Enhances prediction sharpness and prevents degenerate distributions during training.

This formulation simplifies prior diffusion language modeling approaches while maintaining strong generative performance.

---

## Architecture

* Model Type: Diffusion Transformer (DiT-style)
* Parameter Count: **21.26M**
* Tokenization: Standard subword tokenization
* Sequence Length: 128
* Training Strategy: Iterative denoising with time-conditioned modeling

---

## Project Structure

```bash
sddlm/
├── data/
│   └── wikitext2/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── diffusion.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── sample.py
├── quick_train.py
├── test_smoke.py
├── requirements.txt
└── checkpoints/   # excluded from version control
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python3 src/train.py
```

The model is trained using a diffusion-based objective with progressive noise scheduling and time-dependent conditioning.

---

## Evaluation

```bash
python3 src/evaluate.py \
    --checkpoint checkpoints/step_0040000.pt \
    --n_gen 200 \
    --steps 128
```

---

## Results

### Quantitative Performance

| Metric  | Value  |
| ------- | ------ |
| Entropy | 4.0767 |
| Gen PPL | 57.41  |

These results demonstrate competitive generative performance for diffusion-based language modeling, capturing both structural coherence and semantic consistency in generated sequences.

---

## Highlights

* Efficient implementation of **diffusion-based text generation**
* Stable training using **simplified denoising objectives**
* Scalable architecture aligned with modern diffusion frameworks
* Competitive performance across standard evaluation metrics
* Modular and extensible codebase for experimentation

---

## Applications

* Generative language modeling
* Text synthesis and augmentation
* Parallel sequence generation
* Research in diffusion-based NLP models

---

## Future Directions

* Scaling model capacity and training regimes
* Integration with advanced guidance techniques
* Exploration of multi-modal diffusion extensions
* Optimization for faster inference and sampling

---

## Reference

Zhu et al., *Simple Denoising Diffusion Language Models*, 2026.

---

## Author

Kotipalli Venkata Sriram
B.Tech CSE, IIIT Vadodara

---
