# DEPRECATED — SWA-PTIS lane (archived 2026-04-09)

**Do not build on this directory, and do not cite any number from it.**

This folder holds the final state of the **SWA-MPPI → SWA-PTIS** paper lane.
Work stopped on **2026-04-09**; everything here was committed retroactively in
`987b3e3` (2026-07-24) purely to archive it, not because the lane resumed.

## Superseded by DISCA

The live paper is **DISCA** — *"Training-Free Cultural Alignment of Large
Language Models via Persona Disagreement"* — in a **separate repository**:

    cultural_alignment/  →  github.com/trungkiet2005/cultural_alignment
    exp_paper/Paper_New/SWA_DPBR/paper_revised.tex

(`cultural_alignment/` is nested inside this working tree but is its own repo;
it is gitignored here on purpose.)

## Why the numbers here are wrong now

Method naming evolved SWA-MPPI → SWA-PTIS → SWA-DPBR → **DISCA**, and the
evaluation was rebuilt along the way. The two sets of results are **not
comparable — never mix them**:

| | SWA-PTIS (this folder, dead) | DISCA (live) |
|---|---|---|
| Countries | 15 | 20 |
| Backbones | 5 models | 7 backbones / 5 families |
| Main result | JSD −37.6% (Qwen2.5-72B), −34.7% (Llama-3.1-70B) | −10–24% on the six ≥3.8B backbones |
| Small model | — | −3.4% on the 2B backbone |
| Open-ended | not evaluated | −2–7% |

The headline shrank because the claim got honest, not because the method got
worse: broader country/model coverage, an open-ended setting, and the review
rounds documented in `bugs.md` (KL misinterpretation, Prospect-Theory constants
applied to logit², OLS on a bounded probability, the heuristic 35% trigger rate).

## What is still worth reading here

- `bugs.md` — the math/theory critique list that drove the pivot
- `mistral_large_removed.md` — Mistral-Large-2407 numbers and prose stripped
  from the paper, kept so the negative finding is not lost
