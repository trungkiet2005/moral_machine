# SOTA Approaches for Cross-Lingual LLM Moral Alignment — NeurIPS/ICML/ACL Paper

This document surveys methods and framing strategies that are competitive at top-tier venues (NeurIPS, ICML, ICLR, ACL, EMNLP) for research on **LLM value alignment, cultural bias, and cross-lingual moral reasoning**.

---

## 1. Positioning the Paper

Top-tier papers in this area typically combine:
1. **A clear empirical gap** — something the field has not measured or explained
2. **A causal or mechanistic angle** — not just "bias exists" but *why* and *how*
3. **A rigorous evaluation** — multiple models, statistical tests, ablations
4. **An actionable contribution** — a method, dataset, or diagnostic tool

For this project the gap is: *LLM moral preferences are language-dependent in structured, reproducible ways that are not explained by translation quality alone.* The contribution can be empirical + mechanistic + a mitigation method.

---

## 2. Core Methodological Approaches

### 2.1 Contrastive / Controlled Prompting

**What it is:** Present the exact same moral scenario in multiple language conditions and isolate the effect of language from cultural framing.

**Variants:**
- **Language-only control:** Translate scenario en→id, query in id; then translate id→en and query in en → compare
- **Culture-only control:** English prompt but explicitly name a cultural context ("In Indonesian culture, ...")
- **Cross-lingual consistency score:** `|pref(L1) - pref(L2)|` as a metric of alignment instability

**Why it works for top venues:** Controlled experiments are the gold standard. Separating language surface from cultural content is a novel methodological contribution.

**References:** Artetxe et al. (2020) XNLI; Shi et al. (2023) "Language is not all you need"

---

### 2.2 Logit-Level Calibration Analysis

**What it is:** Instead of only extracting the argmax token, extract the raw logits (or log-probabilities) for tokens `"1"` and `"2"`. This gives:
- A continuous preference score instead of binary
- A calibration measure (is the model confident or near-random?)
- A way to detect "forced but uncertain" choices

```python
# Instead of greedy argmax:
logits = model(**inputs).logits[:, -1, :]  # last token position
token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
token_2 = tokenizer.encode("2", add_special_tokens=False)[0]
p1 = logits[:, token_1].softmax(-1)
p2 = logits[:, token_2].softmax(-1)
preference_score = p1 / (p1 + p2)  # continuous [0,1]
```

**Why it works for top venues:** Converts a classification task into a richer probabilistic one. Enables proper statistical testing (e.g., Mann-Whitney U, permutation tests). Calibration analysis is a strong methodological contribution.

**References:** Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"

---

### 2.3 Multi-Model Comparison (Model Family × Language Matrix)

**What it is:** Run the same evaluation across a grid of models:

| | en | de | id | ja | zh-cn | ... |
|--|--|--|--|--|--|--|
| Llama-3.1-8B | | | | | | |
| Llama-3.1-70B | | | | | | |
| Mistral-7B | | | | | | |
| Qwen-2-7B | | | | | | |
| GPT-4o | | | | | | |
| Gemini-1.5 | | | | | | |

Then fit a mixed-effects model: `preference ~ model + language + model×language + (1|scenario)`

**Why it works for top venues:** Reveals whether biases are model-family-specific (training data), architecture-specific, or universal. The interaction term `model×language` is the key finding.

**References:** MultiTP paper; Bender et al. "On the Dangers of Stochastic Parrots"

---

### 2.4 Chain-of-Thought (CoT) Moral Reasoning Extraction

**What it is:** Add `"Explain your reasoning step by step, then answer."` to the prompt. Then:
1. Compare CoT choice vs greedy choice (do they differ?)
2. Extract and classify reasoning patterns (utilitarian / deontological / virtue ethics / refusal)
3. Measure whether CoT reduces cross-lingual variance

**Why it works for top venues:** CoT analysis transforms a black-box bias paper into a mechanistic one. It answers *why* the model chooses differently across languages by surfacing the reasoning chain.

**Coding approach:**
- Parse CoT text with a classifier (a smaller LLM or regex) to label reasoning type
- Compute: `%utilitarian reasoning` vs `%utilitarian choice` — mismatch is evidence of alignment-reasoning disconnect

**References:** Wei et al. (2022) "Chain-of-Thought Prompting"; Kojima et al. (2022) Zero-shot CoT

---

### 2.5 Probing / Representation Analysis

**What it is:** Extract hidden states at the layer where moral reasoning likely occurs and analyze:
- Do representations cluster by language or by moral choice?
- Is there a "moral preference direction" in the residual stream?
- Does this direction rotate across languages?

**Tools:**
- `transformer_lens` for activation extraction
- Linear probe on last-token representation → predict choice
- PCA/t-SNE of scenario representations colored by choice

**Why it works for top venues:** Mechanistic interpretability is a hot area. Showing that language affects the moral representation geometry rather than just surface output is a strong NeurIPS/ICLR claim.

**References:** Marks & Tegmark (2023) "The Geometry of Truth"; Li et al. (2023) "Inference-Time Intervention"

---

### 2.6 Cross-Lingual Alignment Gap Metric (Novel Contribution)

**What it is:** Define a new metric to quantify how much a model's moral preferences shift with language:

```
CLAG(model, scenario_set) =
    (1/|L|^2) Σ_{l1,l2} Σ_d |pref(l1,d) - pref(l2,d)|
```

Where `d` indexes moral dimensions, `l1/l2` are language pairs.

- CLAG = 0: perfectly consistent across languages
- CLAG > 0: language-dependent moral drift

Can also decompose into **within-group** (similar language family) vs **cross-group** (typologically distant) gaps.

**Why it works for top venues:** A well-defined metric with desirable properties (bounded, interpretable, decomposable) is an artifact that the community can reuse. NeurIPS/ICML reviewers reward this.

---

### 2.7 Mitigation: Language-Invariant Moral Prompting

**What it is:** Propose a prompting strategy or lightweight fine-tuning that reduces CLAG:

**Option A — Anchored prompting:**
Prepend a universal moral frame: "Regardless of the language of this scenario, apply consistent ethical principles." Ablate whether this reduces cross-lingual variance.

**Option B — Translate-then-reason:**
For any non-English prompt, instruct the model to first translate internally then reason in English. Measures how much of the bias is language-surface vs deeper.

**Option C — Preference calibration:**
Post-hoc: estimate per-language bias vector from a calibration set and subtract it from logit scores.

**Why it works for top venues:** A paper that only measures a problem is a solid empirical paper. A paper that also proposes and evaluates a mitigation has much stronger NeurIPS/ICML acceptance odds.

---

## 3. Statistical Rigor Requirements

Top venues expect:

| Requirement | How to implement |
|-------------|-----------------|
| Effect size, not just p-value | Cohen's d or rank-biserial r for each dimension × language pair |
| Confidence intervals | Bootstrap 95% CI on preference% (n=460 is sufficient for tight CIs) |
| Multiple comparison correction | Bonferroni or BH for 15 langs × 6 dims = 90 tests |
| Variance decomposition | ANOVA or linear mixed model: `language` vs `dimension` vs `interaction` as explained variance |
| Reproducibility | Seed-fixed greedy decoding (already done in exp01); report exact model hash |

---

## 4. Dataset Contribution Options

| Option | Effort | Impact |
|--------|--------|--------|
| Extend MultiTP to 50+ languages | Medium | High — coverage paper |
| Add human annotation for non-Western langs | High | Very high — gold standard gap |
| Create counterfactual pairs (swap sub1/sub2) | Low | Medium — position-bias control |
| Add cultural expert validation layer | High | Very high — NeurIPS quality |

**Minimum viable:** Extend to 30 languages (add low-resource) + add logit-level scores. This differentiates from MultiTP while building on it.

---

## 5. Paper Structure for NeurIPS/ICML

```
Abstract (150w): gap → method → key finding → implication

1. Introduction (1p)
   - Moral Machine as motivation
   - LLMs deployed globally but aligned monoculturally
   - Our contribution: CLAG metric + cross-lingual analysis + mitigation

2. Related Work (0.5p)
   - MultiTP, Moral Machine (Awad 2018), cross-lingual NLP, value alignment

3. Method (1p)
   - MultiTP benchmark, forced decode, logit extraction, CLAG definition

4. Experiments (2p)
   - 4.1: Cross-lingual preference maps (radar charts)
   - 4.2: CLAG scores per model × language family
   - 4.3: CoT analysis — does reasoning type predict choice?
   - 4.4: Representation probing

5. Mitigation (1p)
   - Prompting ablations
   - CLAG before/after

6. Discussion (0.5p)
   - Which languages/dimensions are most misaligned and why (hypothesis)
   - Limitations: GT translation, benchmark scope

7. Conclusion (0.25p)
```

---

## 6. Competitive Differentiation from MultiTP (ACL 2024)

| Aspect | MultiTP (Naous et al.) | Our Work |
|--------|----------------------|----------|
| Model | GPT-4, LLaMA-2 | LLaMA-3.1-70B + multi-model |
| Languages | 100+ | 15→30+ with low-resource |
| Output | Binary choice | Binary + logit scores (calibration) |
| Analysis | Aggregate % | CLAG metric + per-dim variance |
| Mechanism | None | CoT extraction + probing |
| Mitigation | None | Language-invariant prompting |
| Venue | ACL | NeurIPS / ICML |

---

## 7. Key References to Cite

- Awad et al. (2018) "The Moral Machine experiment" — *Nature*
- Naous et al. (2024) "Having Beer after Prayer? Measuring Cultural Bias in LLMs" — *ACL 2024*
- Benkler et al. (2023) "Assessing Language Model Deployment with Risk Cards" — evaluation methodology
- Santurkar et al. (2023) "Whose Opinions Do Language Models Reflect?" — *ICML 2023*
- Atari et al. (2023) "Which Humans?" — cultural diversity in RLHF
- Rao et al. (2023) "Can LLMs Express Their Uncertainty?" — calibration
- Hendrycks et al. (2021) "Aligning AI With Shared Human Values" — ETHICS benchmark
- Wei et al. (2022) "Chain-of-Thought Prompting" — CoT methodology
- Marks & Tegmark (2023) "The Geometry of Truth" — probing approach
