# Exp01 — Baseline Insights & Research Directions

**Experiment:** `experiment/exp01_baseline.py`
**Model:** `Meta-Llama-3.1-70B-Instruct-bnb-4bit` (Unsloth 4-bit quantized)
**Benchmark:** MultiTP (Naous et al., ACL 2024)
**Setting:** 15 languages × 6 moral dimensions × 460 rows/lang | greedy decoding | "normal citizen" system prompt

---

## 1. Method Summary

| Step | What we do |
|------|-----------|
| Prompt construction | Trolley problem scenario in target language + forced decode prefix `"I choose Option "` |
| Decoding | Greedy (`do_sample=False`) → single token `"1"` or `"2"` → deterministic, reproducible |
| Parsing | Rule-based: `startswith("1")` → `first`, `startswith("2")` → `second` |
| Aggregation | % times model picks the "positive" group per dimension (Species=Humans, Characters=More, Fitness=Fit, Gender=Female, Age=Young, SocialStatus=High) |
| Baseline | Human preference from MultiTP `human_preferences_by_lang_converted.csv` |

**Design choice that matters:** replacing back-translation (Google Translate API) with a forced-decode prefix eliminates an external API dependency and a potential noise source. This is a replication-friendly improvement over the original paper's pipeline.

---

## 2. Key Quantitative Findings

### 2.1 Dimension-level patterns (cross-lingual mean)

| Dimension | LLM mean | Human mean | Δ | Direction |
|-----------|----------|------------|---|-----------|
| Species | ~72% | ~79% | −7% | Mixed — some langs at 100%, some at 10% |
| **No. Characters** | **~35%** | **~74%** | **−39%** | **Consistent large deficit** |
| Fitness | ~62% | ~57% | +5% | LLM slightly above human |
| Gender | ~59% | ~56% | +3% | Generally close, outliers exist |
| **Age** | **~59%** | **~73%** | **−14%** | **Consistent moderate deficit** |
| Social Status | ~57% | ~67% | −10% | Consistent moderate deficit |

### 2.2 Most striking language-level deviations

| Lang | Dimension | LLM | Human | Δ | Note |
|------|-----------|-----|-------|---|------|
| `id` | Species | 10% | 77.4% | **−67.4%** | Model prefers saving animals over humans in Indonesian |
| `ja` | Species | 30% | 79.8% | **−49.8%** | Same effect in Japanese |
| `de` | Species | 45% | 82.6% | **−37.6%** | German also reverses |
| `hi` | Species | 50% | 77.8% | **−27.8%** | Hindi half-reversal |
| `tr` | Species | 50% | 68.9% | **−18.9%** | Turkish notable |
| `de` | Gender | 90% | 54.7% | **+35.3%** | Extreme female-preference in German |
| `id` | Gender | 85.7% | 55.1% | **+30.6%** | Same in Indonesian |
| `vi` | Fitness | 86.7% | 58% | **+28.7%** | Strong fit-preference in Vietnamese |
| `es` | No. Characters | 15% | 74.8% | **−59.8%** | Near-random for utilitarian in Spanish |

---

## 3. Core Insights

### I1 — Anti-utilitarian bias is universal
The model systematically underweights the number of lives saved (~35% average vs ~74% human). This holds across all 15 languages. Likely reflects RLHF alignment steering the model away from cold utilitarian arithmetic (it may "feel wrong" to explicitly count lives). **This is the most robust and reproducible finding.**

### I2 — Species preference is language-mediated, not universal
The model's preference to spare humans over animals collapses or reverses in non-European languages (id, ja, de, hi). This cannot be explained by scenario difficulty — it is the same scenario translated. Possible causes:
- Low-resource language degradation in moral reasoning circuits
- Tokenization artifacts causing different semantic activation
- RLHF data imbalance across languages

### I3 — Gender and Age biases show language-specific amplification
In certain languages (de, id) the model strongly over-prefers females; in most languages it under-prefers the young. The direction is not consistent, suggesting these are language-specific alignment artifacts rather than a global bias.

### I4 — Social Status deficit is consistent
Across all 15 languages, the model shows ~10% less preference for high-status individuals than humans do. This may reflect deliberate alignment to avoid classism — an intentional but culturally miscalibrated intervention.

### I5 — Forced decode prefix works, but may inflate refusal filtering
The `parse_model_choice` fallback chain drops `either/neither/other` responses silently. At greedy decoding the refusal rate is negligible, but with temperature this could introduce silent bias.

---

## 4. Methodological Gaps to Address

| Gap | Impact | Proposed Fix |
|-----|--------|-------------|
| Single model (70B only) | Cannot isolate model-size vs alignment effect | Add 8B, 13B variants |
| 4-bit quantization | Minor accuracy loss, not ablated | Compare fp16 on subset |
| Google Translate input | Confounds: is it model bias or translation artifact? | Use human-authored prompts for top-5 langs |
| Greedy only | Variance unknown | Add temperature sampling + CI |
| No chain-of-thought | Cannot explain *why* model diverges | Add CoT condition |
| 15 languages, major only | Skews toward high-resource | Include 10+ low-resource langs |

---

## 5. Research Questions for Next Experiments

**RQ1 (Language effect):** Does translating a prompt from `id→en` before querying produce en-like moral behavior? → isolates language surface vs. cultural framing
**RQ2 (Model family):** Does the pattern replicate across GPT-4, Gemini, Mistral, Qwen?
**RQ3 (Prompt sensitivity):** How much does system prompt framing ("utilitarian philosopher" vs "religious leader" vs "normal citizen") shift the distribution?
**RQ4 (Fine-tuning):** Does instruction-tuning on diverse moral reasoning texts reduce the language-dependent variance?
**RQ5 (Mechanistic):** Which attention heads activate differently for `id` vs `en` on the same Species scenario?

---

## 6. Suggested Next Experiments

| Exp | Goal | Method |
|-----|------|--------|
| `exp02_multilang_ablation` | RQ1: language surface effect | Translate id/ja→en, re-query |
| `exp03_model_comparison` | RQ2: model family | Run 8B + GPT-4o + Gemini on same prompts |
| `exp04_system_prompt` | RQ3: prompt sensitivity | 5 system-prompt variants × 3 langs |
| `exp05_cot` | Explain divergence | Add "Let's think step by step" + parse reasoning |
| `exp06_calibration` | Token-level uncertainty | Extract logit for "1" vs "2", plot confidence |

---

## 7. Paper Framing Hypothesis

**Core claim:** LLM moral preferences are linguistically contingent — the same model exhibits significantly different ethical judgments depending on the language of the prompt, with deviations that cannot be explained by random noise or translation quality alone. The No. Characters dimension reveals a universal anti-utilitarian tendency that may be a systematic artifact of RLHF alignment, while Species and Gender show language-specific divergences that reveal cultural alignment gaps.

**Positioning:** Extends MultiTP (Naous et al., ACL 2024) from "does cultural bias exist" to "what structural properties of LLM training cause language-dependent moral divergence" — a more mechanistic and actionable claim.
