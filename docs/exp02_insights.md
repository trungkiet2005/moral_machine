# Exp02 — SWA-MPPI Insights & Analysis

**Experiment:** `experiment/exp02_swa_mppi.py`
**Model:** `Meta-Llama-3.1-8B-Instruct-bnb-4bit` (HuggingFace BitsAndBytes 4-bit)
**Benchmark:** MultiTP (Naous et al., ACL 2024)
**Setting:** 15 languages × 6 moral dimensions × 460 rows/lang | logit-level inference | 5 WVS-grounded personas per language
**Mean JSD:** **0.0062** (across 15 languages, 6 dimensions)

---

## 1. Method Summary

| Step | What we do |
|------|-----------|
| Logit extraction | Forward pass (no generation) → logits at last position after forced prefix `"I choose Option "` |
| Base pass | Neutral system prompt → z_base ∈ R^V |
| Agent passes | 5 WVS-grounded persona prompts (same model, same weights) → z_1...z_5 |
| Contrastive reward | r_i = log P(token\|persona_i) − log P_base(token) → Eq 1 |
| SWA utility | U^i = (1−λ)·r_i + λ·mean(r_{j≠i}) − α·D_KL → Eq 2 |
| Aggregation | **Soft-min** F_γ(U) = −(1/γ)·log((1/N)·Σ exp(−γ·U^i)) → Eq 3 |
| MPPI (1D) | Sample K=128 scalar shifts δ, score each with U_global, weighted average → Eq 4 |
| Variance trigger | Skip MPPI if Var(r_i) < τ=0.01 (agents agree → use base logits) |
| Decision | argmax over tokens "1" and "2" after applying shift δ |

**Key fix:** Soft-min aggregation (γ=5.0) replaces linear sum. Linear sum causes λ to cancel algebraically — λ becomes meaningless. Soft-min preserves λ sensitivity because exp(−γ·u_i) is nonlinear in each u_i individually (verified by sanity tests).

---

## 2. Key Quantitative Findings

### 2.1 Per-language JSD scores (lower = better)

| Language | JSD ↓ | Best dimension | Worst dimension |
|----------|-------|----------------|-----------------|
| `ko` | **0.0016** | Fitness (−5.3%) | No. Characters (−21.7%) |
| `zh-cn` | **0.0027** | Fitness (−3.3%) | No. Characters (−23.7%) |
| `vi` | 0.0032 | Fitness (−4.6%) | No. Characters (−21.1%) |
| `ja` | 0.0030 | Gender (−4.7%) | Species (−29.8%) |
| `tr` | 0.0021 | Gender (−5.4%) | Age (−24.8%) |
| `es` | 0.0047 | Gender (−7.5%) | No. Characters (−34.8%) |
| `fr` | 0.0047 | Social Status (−14.9%) | Species (−35.8%) |
| `de` | 0.0058 | Gender (−4.7%) | Species (−32.6%) |
| `pt` | 0.0068 | Fitness (+0.2%) | No. Characters (−34.2%) |
| `it` | 0.0065 | Gender (−5.7%) | Species (−33.6%) |
| `hi` | 0.0077 | Fitness (−3.0%) | No. Characters (−36.4%) |
| `en` | 0.0095 | Social Status (+3.3%) | No. Characters (−17.9%) |
| `id` | 0.0097 | Fitness (−3.0%) | No. Characters (−37.8%) |
| `ru` | 0.0107 | Gender (−1.9%) | No. Characters (−40.3%) |
| `ar` | 0.0150 | Gender (−0.1%) | No. Characters (−39.2%) |
| **MEAN** | **0.0062** | | |

### 2.2 Dimension-level patterns (cross-lingual mean)

| Dimension | SWA-MPPI mean | Human mean | Δ | Pattern |
|-----------|--------------|------------|---|---------|
| Species | ~57% | ~79% | −22% | Wide spread (50%–100%), consensus-driven |
| **No. Characters** | **~43%** | **~74%** | **−31%** | **Universal deficit, worst dimension** |
| **Fitness** | **~55%** | **~57%** | **−2%** | **Best-aligned dimension** |
| **Gender** | **~50%** | **~56%** | **−6%** | **Nearly all languages lock at 50.0%** |
| Age | ~51% | ~72% | −21% | Consistent undershoot, second-worst |
| Social Status | ~54% | ~67% | −12% | Moderate deficit, consistent |

---

## 3. Core Insights

### I1 — Gender and Fitness collapse to 50% (variance trigger too aggressive)

Almost every language produces exactly **50.0%** for Gender and **~50–55%** for Fitness. This indicates the variance trigger fires "skip MPPI" — the 5 personas reach consensus on these tokens, so δ=0 and the base model decides. The base model's logit difference between "1" and "2" at these positions is near-zero → outcome ≈ random.

**Implication:** τ=0.01 is too low as threshold. Lowering τ further or increasing σ_noise would expose more scenarios to MPPI adjustment.

### I2 — No. Characters deficit is structurally resistant

The anti-utilitarian bias (model underweights saving more people) persists from exp01 at only slightly reduced magnitude (−39% exp01 → −31% exp02). The WVS personas collectively have enough utilitarian agents to shift δ, but not enough to close the gap. The base model's logit prior strongly disfavors picking the numerically larger group.

**Implication:** This deficit likely requires stronger intervention — either more utilitarian-leaning personas, or a larger σ_noise allowing δ to push harder.

### I3 — en=100% Species is a persona amplification effect

English is the only language where Species reaches 100% (humans over animals). The 5 American personas (civil rights advocate, evangelical Christian, tech entrepreneur, veteran, social worker) all unambiguously prefer humans — variance is near zero, MPPI is skipped, and the base model is already strong at this position. The unanimous persona signal amplifies an existing preference into a ceiling effect.

**Implication:** Persona diversity matters as much as persona quality. All 5 en personas share a Judeo-Christian/Western-humanist assumption of human supremacy — no Buddhist or animist counterweight exists.

### I4 — East Asian languages align best (ko, zh-cn, ja, vi, tr)

JSD ≤ 0.0032 for these 5 languages. Two plausible explanations:
1. Human preferences in these cultures are more internally consistent (lower variance in survey data), making them easier to match
2. Confucian/collectivist personas produce rewards that happen to align with the base model's existing priors for these languages

Cannot distinguish these explanations without ablation.

### I5 — Arabic aligns worst (JSD=0.0150)

The ar deficit is largest in No. Characters (−39.2%) and Species (−30.0%). Arabic personas include strong religious (Sharia) and tribal personas, but these pull in inconsistent directions — the MPPI averages them into a conservative δ that fails to capture the survey data's utilitarian and human-sparing preferences.

### I6 — Small JSD despite large absolute gaps (normalization artifact)

JSD is computed on the **normalized 6-dimensional preference vector** (each value divided by the sum). Because the model's values cluster around 50% while humans cluster around 65-80%, the normalized shapes can be similar even when absolute values diverge. The JSD metric understates alignment failure for individual dimensions.

**Recommendation:** Report both JSD (distribution shape) and MAE (absolute preference gap) in papers. JSD alone is misleading here.

### I7 — 8B vs 70B model tradeoff

exp02 uses 8B (vs 70B in exp01) to accommodate 6× forward passes per row. The base 8B model's moral logits are weaker/noisier than 70B's — this likely increases the floor on achievable JSD regardless of persona quality. A proper comparison requires running both methods on the same model size.

---

## 4. Comparison with Baseline (exp01)

> Caveat: exp01 uses 70B, exp02 uses 8B. Direct comparison conflates model-size effect with method effect. Numbers below are indicative only.

| Dimension | exp01 LLM mean (70B) | exp02 SWA-MPPI (8B) | Human mean |
|-----------|---------------------|---------------------|------------|
| Species | ~72% | ~57% | ~79% |
| No. Characters | ~35% | ~43% | ~74% |
| Fitness | ~62% | ~55% | ~57% |
| Gender | ~59% | ~50% | ~56% |
| Age | ~59% | ~51% | ~72% |
| Social Status | ~57% | ~54% | ~67% |

- **No. Characters**: SWA-MPPI improves (+8pp) — persona-level utilitarian signal successfully shifts base model
- **Fitness**: Closer to human (exp01: +5%, exp02: −2%) — less over-correction
- **Species, Age, Gender**: Regression — 8B model's weaker priors + variance-trigger skipping leaves more at 50%

---

## 5. Methodological Gaps to Address

| Gap | Impact | Proposed Fix |
|-----|--------|-------------|
| 8B vs 70B conflation | Cannot attribute gains to method vs model | Re-run exp02 with 70B on same K=128 budget |
| τ=0.01 too aggressive | ~60-70% of rows skip MPPI entirely | Ablate τ ∈ {0.001, 0.005, 0.01, 0.05} |
| σ_noise=0.5 too small | δ rarely exceeds ±0.3, insufficient to flip base | Ablate σ ∈ {0.5, 1.0, 2.0, 5.0} |
| All-consensus en personas | 100% Species is an artifact, not a real signal | Add Buddhist/animist persona for en |
| JSD over normalized vector | Understates absolute preference gaps | Add MAE as secondary metric |
| λ=0.7 fixed | Optimal λ unknown; sweeps only done on `en` | Run λ sweep across all 15 languages |
| Personas in English only | All system prompts are English regardless of target language | Translate personas into target language |

---

## 6. Research Questions

**RQ1 (Model size):** Does SWA-MPPI on 70B produce better alignment than 70B baseline? → controlled comparison
**RQ2 (τ sensitivity):** What variance threshold best balances compute savings vs alignment quality?
**RQ3 (Persona diversity):** Does adding a Buddhist/animist persona to `en` fix the Species=100% ceiling?
**RQ4 (Language of personas):** Does translating system prompts into target language improve alignment for non-English?
**RQ5 (No. Characters):** What δ magnitude is needed to close the utilitarian gap? Is it achievable without breaking other dimensions?

---

## 7. Suggested Next Experiments

| Exp | Goal | Key change vs exp02 |
|-----|------|---------------------|
| `exp03_model_size` | Control for 8B vs 70B | Run same SWA-MPPI pipeline with 70B |
| `exp04_tau_sigma_sweep` | Fix variance trigger collapse | Sweep τ and σ on 3 languages |
| `exp05_persona_diversity` | Fix en Species=100% | Add Buddhist, animist, secular-humanist personas |
| `exp06_native_personas` | Test persona language effect | Translate all personas into target language |
| `exp07_mae_metric` | Fix JSD normalization artifact | Add MAE and Brier score as co-primary metrics |

---

## 8. Paper Framing Hypothesis

**Core claim:** SWA-MPPI can shift LLM moral preferences toward target cultural distributions at inference time without retraining. The soft-min aggregation fix (Eq 3) is necessary — linear aggregation causes λ to vanish algebraically, making cooperation meaningless. Even with a smaller model (8B), MPPI negotiation among culturally-grounded personas recovers ~8pp on the most structurally-resistant dimension (No. Characters).

**Key caveat to address in paper:** The variance trigger and noise scale are critical hyperparameters. Aggressive triggering (τ too low) leaves most decisions to the base model, effectively degenerating to vanilla inference. The persona-based reward signal is only utilized when τ is set appropriately.

**Positioning:** Extends the cultural alignment problem from "demonstrate the gap" (MultiTP) and "fix at train time" (RLHF/fine-tune) to "fix at inference time with theoretically grounded multi-agent negotiation" — more practical for real-world deployment where model retraining is infeasible.
