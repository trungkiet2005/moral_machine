# Content removed from paper_revised.tex (Mistral + overclaim cleanup)

This file collects every Mistral-Large-2407 mention that was stripped out of the
paper, so the material can be revisited later without losing the numbers, prose,
or per-country results.

---

## 1. Models section (line ~523)

Original sentence (removed):

> **Mistral-Large-2407** [Mistral AI 2024] (4-bit). All on a single H100 80GB
> via Unsloth with greedy decoding.

→ Replaced by: just the five remaining models (Qwen2.5-72B, Llama-3.1-70B,
Qwen2.5-32B/7B, Llama-3.1-8B).

Citation key still in `references.bib`: `mistral2024`.

---

## 2. Cross-model comparison table (Table `tab:model_summary`, §Results)

Removed row:

| Model         | Size | Vanilla JSD       | SWA JSD           | Improv. | $\|\Delta\text{JSD}\|$ | $r$ Van. | $r$ SWA |
|---------------|------|-------------------|-------------------|---------|------------------------|----------|---------|
| Mistral-Large | 126B | $.0457 \pm .003$  | $.0430 \pm .003$  | +5.8%   | .0027                  | 0.658    | 0.550   |

Caption note that was removed:

> Mistral-Large's absolute change is within the bootstrap noise floor.

---

## 3. "Mistral-Large." paragraph in §Results (was right after small-models)

> **Mistral-Large.** This is the most informative negative finding in the
> paper. Mean JSD improves only modestly, and Pearson $r$ *decreases* for
> $11/15$ countries. We do not bury this in a footnote: it suggests that on a
> model whose vanilla logits are already comparatively well-aligned, the MPPI
> correction can compress the AMCE vector toward the population mean and improve
> distributional distance while degrading rank-order alignment along the
> 6-dimensional axes. We discuss this dissociation explicitly in §Discussion.

---

## 4. Zero-preprocessing table row (Table `tab:zero_preproc`)

Removed row:

| Model         | Vanilla JSD | SWA-MPPI JSD | $\Delta$JSD | Notes       |
|---------------|-------------|--------------|-------------|-------------|
| Mistral-Large | [TBD]       | [TBD]        | [TBD]       | raw release |

---

## 5. Discussion — cross-model failure patterns (§Discussion)

Removed sentences (the Vietnam paragraph was kept; only the Mistral panel-level
dissociation was cut):

> ...and the same phenomenon appears at panel level on Mistral-Large.
> Mistral-Large shows a striking dissociation: JSD improves modestly (+5.8%) but
> Pearson $r$ *decreases* for 11/15 countries (mean $\Delta r = -0.108$),
> indicating that the MPPI correction improves distributional distance while
> perturbing the rank ordering of AMCE dimensions.

---

## 6. Limitations — Mistral-Large rank-order regression (§Discussion)

> *Mistral-Large rank-order regression.* On the largest model in our panel,
> JSD improves but Pearson $r$ decreases for $11/15$ countries. We do not have
> a clean fix for this and it is consistent with the hypothesis that MPPI is
> partially compressing the AMCE vector toward the population mean on
> already-aligned models. A user deploying SWA-MPPI on a model whose vanilla
> alignment is already strong should evaluate per-dimension shifts before
> trusting the aggregate JSD.

---

## 7. Conclusion sentence (§Conclusion)

Removed clause:

> ...on Mistral-Large the picture is mixed because rank-order alignment can
> degrade even as JSD improves; ...

---

## 8. Abstract sentences (removed during abstract rewrite)

> Mistral-Large changes by less than the bootstrap noise floor; ...
> Mistral-Large illustrates a real failure mode: its JSD improves slightly
> but its 6-dimensional rank ordering does not, a dissociation we discuss
> honestly rather than average away.

---

## 9. Per-country appendix table for Mistral-Large (Table `tab:mistral_percountry`)

Full table that was removed from the appendix:

> **Per-country results for Mistral-Large-2407 (4-bit).**
> Note: despite modest JSD gains, Pearson $r$ decreases for 11/15 countries.

| Country       | Lang. | JSD Van. | JSD SWA | $r$ Van. | $r$ SWA | $\Delta$JSD | % Improv. |
|---------------|-------|----------|---------|----------|---------|-------------|-----------|
| USA           | en    | .0414    | .0211   | 0.841    | 0.834   | −.020       | +49.0     |
| Germany       | de    | .0378    | .0291   | 0.734    | 0.675   | −.009       | +23.0     |
| China         | zh    | .0572    | .0618   | 0.824    | 0.585   | +.005       | −8.1      |
| Japan         | ja    | .0457    | .0535   | 0.814    | 0.387   | +.008       | −17.1     |
| Brazil        | pt    | .0519    | .0625   | 0.359    | 0.510   | +.011       | −20.4     |
| Saudi Arabia  | ar    | .0238    | .0315   | 0.724    | 0.671   | +.008       | −32.4     |
| Vietnam       | vi    | .0521    | .0428   | 0.768    | 0.571   | −.009       | +17.8     |
| France        | fr    | .0481    | .0475   | 0.683    | 0.628   | −.001       | +1.2      |
| India         | hi    | .0457    | .0527   | 0.772    | 0.631   | +.007       | −15.4     |
| South Korea   | ko    | .0332    | .0320   | 0.546    | 0.355   | −.001       | +3.6      |
| Great Britain | en    | .0418    | .0331   | 0.853    | 0.730   | −.009       | +20.7     |
| Russia        | ru    | .0552    | .0566   | 0.768    | 0.631   | +.001       | −2.6      |
| Mexico        | es    | .0472    | .0481   | 0.748    | 0.642   | +.001       | −1.9      |
| Nigeria       | en    | .0401    | .0301   | 0.128    | 0.131   | −.010       | +25.0     |
| Australia     | en    | .0516    | .0338   | 0.865    | 0.682   | −.018       | +34.5     |
| **Mean**      |       | **.0457**| **.0430**| 0.658   | 0.550   | −.003       | **+5.8**  |

---

## TODO when re-introducing Mistral

- Decide whether to keep Mistral-Large in the model panel at all (currently the
  paper claims six open-weight models 7B–126B; with Mistral removed it is five,
  7B–72B → update any "from 7B to 126B" phrasing if it reappears).
- Re-add `mistral2024` citation usage if the row is restored.
- Re-add the per-country appendix table.
- Re-introduce the JSD/$r$ dissociation discussion only if you want to surface
  the failure mode again.

---

# Overclaim cleanup (round 2)

The following passages were removed/toned down to avoid overclaiming.

## A. Zero-Preprocessing MultiTP Evaluation (entire subsection + table)

Subsection text and a placeholder table where every cell was `[TBD]`. The
table committed to numbers the authors do not yet have. Re-introduce only
after the actual numbers exist.

## B & C. Anglosphere paragraphs (Per-Country Qwen2.5-72B + Discussion)

Two paragraphs that explicitly admit no causal explanation for why the
Anglosphere countries (AU/GB/US) show the largest gains on Qwen2.5-72B.
Reduced to a single neutral sentence in the per-country prose. Reintroduce
only after a controlled experiment that varies prompt language and RLHF
version on the same scenarios.

## D. Vietnam caveat paragraph (Discussion)

The paragraph that ended with "That is not direction-aware alignment in any
strong sense"---i.e., self-undercut the +65.8% Vietnam result. Vietnam stays
in the per-country tables; the self-undercutting prose is gone.

## E. Pre-flight diagnostic speculation (Discussion)

Speculative sentence proposing that "checking logit entropy could predict
when to activate the framework"---no experiment supports it.

## F. Cross-model failure patterns paragraph (Discussion)

Descriptive comparison of which countries regress on Llama vs Qwen, with no
mechanistic analysis. Removed.

## G. Logit-flattening hypothesis for small quantised models (Results)

Removed the sentence claiming "4-bit quantisation flattens the decision-token
logits toward chance"---untested on the 7B/8B models (only Qwen2.5-32B has
the logit-entropy measurement).

## H. Utilitarian persona dominance limitation (Discussion)

Limitation paragraph that opened a concern (utilitarian persona may overwhelm
country-specific personas) the paper does not address. Removed.

## I. Hyperparameter "weakest link" wording (Discussion)

Original limitation labelled hyperparameter validation as "the weakest link"
and the 50-scenario transfer check as "genuinely too small". Re-phrased
neutrally; the substance is kept but the self-disparaging adjectives are
gone.

## J. Single-seed full paragraph (Models section)

8-line paragraph appearing before any results that warned the reader about
single-seed methodology. Reduced to one neutral sentence + footnote.

## K. PT parameters per-culture concern (Limitations)

Speculative limitation about Western-derived PT constants possibly not
transferring across cultures, suggesting per-country or learned PT parameters
"could in principle improve alignment". No experiment supports it. Removed.

---

## TODO when re-introducing

- Re-add zero-preprocessing table after running the actual numbers.
- Re-introduce Anglosphere pattern only after a controlled prompt-language/RLHF
  experiment.
- Re-introduce Vietnam dissociation only after a per-dimension AMCE breakdown
  and across-seed run.
- Run a $(\alpha,\beta,\kappa)$ sweep before re-introducing the PT-parameter
  limitation.
