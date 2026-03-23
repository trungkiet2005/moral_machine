# Exp03 — SWA-MPPI-v2 Failure Analysis

**Experiment:** `experiment/exp03_swa_mppi_v2.py`
**Model:** `Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Status:** FAILED — all predictions collapsed to 50.0% for every language/dimension
**Mean JSD:** 0.0032 (misleadingly low — artifact of 50% collapse, not real alignment)
**Mean MAE:** 17.56pp (honest metric — reveals the failure)

---

## 1. Symptom

Every language, every dimension → `v2 = 50.0%`. Sample outputs confirm the model
always returns raw `'1'`, regardless of scenario content:

```
sub1=Animals  sub2=Humans  → choice=first ('1')   # saves Animals ✗
sub1=Humans   sub2=Animals → choice=first ('1')   # saves Humans ✓
```

Because the dataset has balanced sub1/sub2 ordering, always choosing '1' gives
exactly 50% for every dimension. MPPI trigger rate was 100% (FIX 1 worked),
but the MPPI itself was producing a pathological result.

---

## 2. Root Cause: Broken MPPI Reward Formula

### The bug (present in exp02 and exp03)

In `mppi_step_1d`, the "effective reward" for agent i at shift δ was computed as:

```python
r1 = agent_rewards_binary[:, 0] + delta_k   # r_i^(1) + δ
r2 = agent_rewards_binary[:, 1] - delta_k   # r_i^(2) - δ
effective_rewards = r1 - r2                  # (r_i^(1) - r_i^(2)) + 2δ
```

This is **linear in δ**:
```
eff_i(δ) = (r_i^(1) - r_i^(2)) + 2δ
```

The KL penalty was `α · 0.5 · δ²` with α=0.05 → `0.025 · δ²`.

The aggregate objective is approximately:
```
U_global(δ) ≈ C + 2δ - 0.025·δ²
```
Optimal: `d/dδ = 2 - 0.05·δ = 0` → `δ* = 40.0` — a massive positive shift.

**MPPI always found δ >> 0 → always boosted option 1's logit to dominate → always chose '1'.**

### Why exp02 didn't show this

In exp02, `tau_conflict=0.01` was too high — MPPI was **almost never triggered**
(trigger rate ≈ 0%, though not logged). Results in exp02 were driven by the base
model's raw logit comparison (`argmax(z_base[[id_1, id_2]])`), not by MPPI.
This produced some variation (not all 50%), but reflected base model priors, not
cultural alignment.

The broken MPPI formula was hidden in exp02. Fixing τ in exp03 exposed it.

### Timeline of bugs

| Exp | τ | MPPI fires? | MPPI formula | Result |
|-----|---|-------------|-------------|--------|
| exp02 | 0.01 (too high) | ~0% | Broken (linear δ) | Base model decides; ~50% for some dims, variation for others |
| exp03 | 0.001 | 100% | Broken (linear δ) | δ→∞ → always "1" → exactly 50% everywhere |
| **exp04** | **0.001** | **100%** | **Fixed (expected reward)** | **Correct negotiation** |

---

## 3. Correct MPPI Formulation

### What the formula should be

When shift δ is applied, the decision probabilities change:
```
p(δ) = softmax([z_base[id_1] + δ,  z_base[id_2] - δ])
```

Agent i's **expected** contrastive reward under this shifted distribution is:
```
eff_i(δ) = p₁(δ) · r_i^(1) + p₂(δ) · r_i^(2)
```

As δ → +∞: p₁ → 1, eff_i → r_i^(1)  (always choose "1")
As δ → -∞: p₂ → 1, eff_i → r_i^(2)  (always choose "2")
At δ = 0:  eff_i = base-weighted average of r_i^(1) and r_i^(2)

The KL penalty should be the true KL divergence:
```
D_KL = Σ_j p_j(δ) · log( p_j(δ) / p_j(0) )
```

This creates a proper interior optimum: MPPI negotiates between agent preferences
while being regularized by how far the shift moves from the base distribution.

### Why this fixes the collapse

- `eff_i(δ)` is bounded: `min(r_i^(1), r_i^(2)) ≤ eff_i(δ) ≤ max(r_i^(1), r_i^(2))`
- KL grows unboundedly as δ → ±∞ → δ* stays finite
- If agents disagree (some prefer r^(1), others r^(2)), the optimal δ reflects the negotiated compromise

---

## 4. Code Review: All Bugs Found Across exp01–exp03

### Bug B1 — MPPI linear reward [exp02, exp03, CRITICAL]
**Location:** `mppi_step_1d()`
**Issue:** `eff = (r1 + δ) - (r2 - δ)` is linear in δ → unbounded optimum
**Fix:** `eff_i = softmax([l1+δ, l2-δ])[0] * r1 + softmax([l1+δ, l2-δ])[1] * r2`
**Impact:** All MPPI-fired decisions were wrong in both exp02 (rare) and exp03 (always)

### Bug B2 — KL approximation invalid at large σ [exp03]
**Location:** `kl_approx = 0.5 * delta_k ** 2`
**Issue:** Second-order Taylor approximation of KL only valid for δ << 1.
With σ=1.5, most samples have |δ| > 1 → approximation breaks down
**Fix:** Compute true binary KL: `Σ p_j(δ) log(p_j(δ)/p_j(0))`
**Impact:** Even with correct reward formula, large δ was under-penalized

### Bug B3 — τ too high in exp02 [exp02, MODERATE]
**Location:** `tau_conflict = 0.01`
**Issue:** Almost all rows skipped MPPI → personas had no effect → results = base model
**Fix:** τ=0.001 (exp03 fixed this, but exposed B1)
**Impact:** exp02 results reflect 8B base model biases, not SWA-MPPI

### Bug B4 — base_logits not passed to MPPI [exp02, exp03]
**Location:** `mppi_step_1d()` signature
**Issue:** MPPI does not know where the base distribution is → cannot compute p(δ) or true KL
**Fix:** Add `base_logits_binary: torch.Tensor` parameter

### Bug B5 — JSD over normalized vector understates gaps [exp02, exp03, METRIC]
**Location:** `compute_jsd()`
**Issue:** Normalizing [50, 50, 50, 50, 50, 50] and [80, 74, 57, 56, 72, 66]
produces similar-shaped distributions → small JSD despite 18pp MAE
**Fix:** Report MAE alongside JSD (exp03 added this; exp04 keeps it)

---

## 5. What exp04 Fixes

1. **B1 + B2 + B4**: Rewrite `mppi_step_1d` with correct expected-reward + true KL
2. **B3**: Keep τ=0.001 (already correct in exp03)
3. All other exp03 improvements retained (6 personas, MAE metric, trigger logging)
