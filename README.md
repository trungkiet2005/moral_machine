# MultiTP: LLM Moral Alignment in Multilingual Trolley Problems

Replication and extension of **"Language Model Alignment in Multilingual Trolley Problems"** (Jin et al., ICLR 2025).

This project investigates whether LLM moral preferences are **language-dependent** in structured, reproducible ways not explained by translation quality alone — tested across **15 languages** and **6 moral dimensions**.

---

## Research Questions

| RQ | Description |
|----|-------------|
| RQ1 | **Global alignment** — MIS score (L2 distance of 6D preference vectors vs. human baseline) |
| RQ2 | **Per-dimension decomposition** — preference breakdown + radar charts |
| RQ3 | **Language sensitivity** — std dev across languages + K-means clustering (k=4) |
| RQ4 | **Language inequality** — Pearson correlation(MIS, #speakers) |
| RQ5 | **Prompt paraphrase robustness** — 5 paraphrases, Fleiss' Kappa, pairwise F1 |

---

## Experiments

| Experiment | Method | Key Idea |
|------------|--------|----------|
| `exp01_baseline.py` | Greedy decoding | Baseline token-forcing, counterbalanced scenarios |
| `exp02_swa_mppi.py` | SWA-MPPI v1 | Sliding-window agreement + multi-persona preference inference |
| `exp03_swa_mppi_v2.py` | SWA-MPPI v2 | Bug fix: formula correction for preference aggregation |
| `exp04_swa_mppi_v3.py` | SWA-MPPI v3 | Further refinement of MPPI weighting |
| `exp05_cot_reasoning.py` | Chain-of-Thought | 2-pass inference: reason first, then force choice |
| `exp06_translate_then_reason.py` | Translate-then-Reason | Translate scenario to English → reason → translate answer back |
| `exp07_logit_calibration.py` | Logit Calibration | Post-hoc: estimate per-language bias vector, subtract from logits |
| `exp08_full_paper_replication.py` | Full Replication | All 5 RQs, exact methodology from Jin et al. 2025 |

---

## Models Tested

- `Llama-3.1-70B-Instruct` (4-bit quantized, primary)
- `Llama-3.1-8B-Instruct` (4-bit)
- `Qwen2.5-72B-Instruct`
- `Qwen2.5-32B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Gemma-2-27B`
- `Mistral-Large`
- `CohereForAI/c4ai-command-r-plus`

---

## Moral Dimensions

6 dimensions from the original Moral Machine experiment:

| Dimension | Description |
|-----------|-------------|
| Species | Humans vs. animals |
| Gender | Male vs. female |
| Fitness | Athletic vs. large body type |
| Social Status | High-status vs. homeless |
| Age | Young vs. elderly |
| No. Characters | More lives vs. fewer lives (utilitarian) |

---

## Setup & Usage

### Requirements
```bash
pip install unsloth torch transformers tqdm pandas matplotlib scipy scikit-learn
```

### Run on Kaggle (recommended)
The experiments are designed for **Kaggle GPU** (T4 / P100 / A100):

1. Upload the `MultiTP/` dataset folder to Kaggle
2. Set your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
3. Run any experiment notebook/script

> **Security note:** Never hardcode API tokens in source files. Use environment variables or secret managers.

### Run locally
```bash
export HF_TOKEN=your_token_here
python experiment/exp01_baseline.py
python experiment/exp08_full_paper_replication.py
```

---

## Key Findings

- LLMs show **systematic, language-dependent** moral biases not reducible to translation artifacts
- The **No. Characters** dimension shows the largest deviation from human preferences (~−39% in baseline)
- **CoT reasoning** shifts choices toward more utilitarian outcomes
- **Logit calibration** (exp07) achieves the best alignment with human preferences by correcting per-language bias vectors

---

## Reference

```bibtex
@inproceedings{jin2025multitp,
  title     = {Language Model Alignment in Multilingual Trolley Problems},
  author    = {Jin, Zhijing and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

Paper: [arXiv 2407.02273](https://arxiv.org/abs/2407.02273)
