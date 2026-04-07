# Baseline Comparison Tracker (Paper-Ready)

## Scope
- Dataset/protocol: MultiTP per-country native-language evaluation, 15 countries, 342 scenarios/country.
- Main method (updated): `SWA-MPPI v3` on `Llama-3.1-70B-Instruct`.
- Baselines:
  - `B1`: Country-Tailored Instruction Prompt (`Llama-3.1-70B-Instruct-bnb-4bit`)
  - `B2`: WVS Profile Prompting (`Llama-3.1-70B-Instruct-bnb-4bit`)
  - `B3`: PRISM-Style Pluralistic Prompting (`Llama-3.1-70B-Instruct-bnb-4bit`)
  - `Vanilla (Llama)`: same backbone as SWA, no SWA/MPPI control
- Additional method study (not a baseline):
  - `Activation Steering (AS-0)`: single-pass CAA steering (`target_layer=40`, `steering_strength=1.5`, `n_calibration=30`)

## Table A — Aggregate Comparison (Main Method vs All Baselines)

| Method | Mean JSD (↓) | Mean Pearson r (↑) | Delta JSD vs SWA | Relative Gap vs SWA |
|---|---:|---:|---:|---:|
| **SWA-MPPI v3 (Main, Llama-70B)** | **0.0612** | **0.3840** | 0.0000 | 0.0% |
| Vanilla (Llama-70B) | 0.0937 | -0.0260 | +0.0325 | +34.7% |
| Activation Steering (AS-0, method) | 0.0877 | -0.2084 | +0.0265 | +30.2% |
| B1: Country-Tailored Instruction Prompt | 0.0680 | 0.2327 | +0.0068 | +11.1% |
| B3: PRISM-Style Pluralistic Prompting | 0.0699 | 0.1475 | +0.0087 | +12.5% |
| B2: WVS Profile Prompting | 0.0737 | 0.1554 | +0.0125 | +16.9% |

**Takeaway:** by JSD (primary metric), `SWA-MPPI v3` is best overall.

## Table B — Per-Country JSD Full Comparison (with Steering Method)

| Country | B1 | B2 | B3 | Steering (AS-0) | Vanilla (Llama) | SWA (Llama) | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| USA | 0.1694 | 0.1414 | **0.1074** | 0.1565 | 0.1561 | 0.1234 | B3 |
| DEU | 0.0501 | 0.0482 | 0.0556 | 0.0656 | 0.0709 | **0.0391** | SWA |
| CHN | 0.0516 | 0.0356 | 0.0499 | 0.0488 | 0.0459 | **0.0311** | SWA |
| JPN | 0.0442 | 0.0432 | **0.0376** | 0.0460 | 0.0457 | 0.0416 | B3 |
| BRA | 0.0552 | **0.0430** | 0.0598 | 0.0570 | 0.0532 | 0.0656 | B2 |
| SAU | 0.0406 | **0.0351** | 0.0545 | 0.0596 | 0.0539 | 0.0596 | B2 |
| VNM | 0.0557 | 0.0784 | 0.0695 | 0.0892 | 0.1546 | **0.0529** | SWA |
| FRA | **0.0463** | 0.0470 | 0.0744 | 0.0542 | 0.0500 | 0.0516 | B1 |
| IND | 0.0443 | 0.0464 | 0.0499 | 0.0494 | 0.0442 | **0.0337** | SWA |
| KOR | 0.0777 | 0.0715 | 0.0578 | 0.0700 | 0.0893 | **0.0436** | SWA |
| GBR | 0.0967 | 0.1280 | 0.1015 | 0.1589 | 0.1568 | **0.0918** | SWA |
| RUS | 0.0456 | 0.0708 | 0.0537 | 0.0525 | 0.0449 | **0.0413** | SWA |
| MEX | **0.0384** | 0.0718 | 0.0739 | 0.0836 | 0.1265 | 0.0425 | B1 |
| NGA | 0.1006 | 0.1179 | **0.0988** | 0.1642 | 0.1582 | 0.0988 | B3/SWA tie |
| AUS | 0.1033 | 0.1275 | 0.1036 | 0.1600 | 0.1551 | **0.1010** | SWA |

## Table C — SWA Llama Delta Table (as provided)

| Country | Van. JSD | SWA JSD | Delta JSD | Van. Pearson | SWA Pearson | Delta Pearson | JSD Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA | 0.1561 | 0.1234 | -0.033 | 0.012 | 0.197 | +0.185 | +20.9% |
| DEU | 0.0709 | 0.0391 | -0.032 | -0.455 | 0.685 | +1.140 | +44.9% |
| CHN | 0.0459 | 0.0311 | -0.015 | 0.406 | 0.802 | +0.396 | +32.3% |
| JPN | 0.0457 | 0.0416 | -0.004 | 0.496 | 0.466 | -0.030 | +9.0% |
| BRA | 0.0532 | 0.0656 | +0.012 | -0.370 | 0.348 | +0.718 | -23.3% |
| SAU | 0.0539 | 0.0596 | +0.006 | 0.043 | -0.210 | -0.253 | -10.7% |
| VNM | 0.1546 | 0.0529 | -0.102 | -0.519 | -0.135 | +0.384 | +65.8% |
| FRA | 0.0500 | 0.0516 | +0.002 | 0.250 | 0.394 | +0.143 | -3.0% |
| IND | 0.0442 | 0.0337 | -0.010 | -0.315 | 0.715 | +1.030 | +23.6% |
| KOR | 0.0893 | 0.0436 | -0.046 | -0.272 | 0.551 | +0.823 | +51.2% |
| GBR | 0.1568 | 0.0918 | -0.065 | 0.012 | 0.286 | +0.274 | +41.5% |
| RUS | 0.0449 | 0.0413 | -0.004 | 0.539 | 0.637 | +0.098 | +7.9% |
| MEX | 0.1265 | 0.0425 | -0.084 | -0.371 | 0.307 | +0.677 | +66.4% |
| NGA | 0.1582 | 0.0988 | -0.059 | 0.109 | 0.497 | +0.388 | +37.5% |
| AUS | 0.1551 | 0.1010 | -0.054 | 0.039 | 0.224 | +0.185 | +34.9% |
| **Mean** | **0.0937** | **0.0612** | **-0.033** | **-0.026** | **0.384** | **+0.411** | **+34.7%** |

## Table D — Method Study: Activation Steering (AS-0) vs Vanilla (Llama)

| Country | Vanilla JSD | Steering JSD | Delta JSD (Steering - Vanilla) |
|---|---:|---:|---:|
| USA | 0.1615 | 0.1565 | -0.0050 |
| DEU | 0.0713 | 0.0656 | -0.0057 |
| CHN | 0.0424 | 0.0488 | +0.0064 |
| JPN | 0.0396 | 0.0460 | +0.0063 |
| BRA | 0.0585 | 0.0570 | -0.0014 |
| SAU | 0.0512 | 0.0596 | +0.0084 |
| VNM | 0.1390 | 0.0892 | -0.0498 |
| FRA | 0.0520 | 0.0542 | +0.0022 |
| IND | 0.0421 | 0.0494 | +0.0073 |
| KOR | 0.0735 | 0.0700 | -0.0035 |
| GBR | 0.1622 | 0.1589 | -0.0032 |
| RUS | 0.0468 | 0.0525 | +0.0057 |
| MEX | 0.1175 | 0.0836 | -0.0339 |
| NGA | 0.1628 | 0.1642 | +0.0014 |
| AUS | 0.1606 | 0.1600 | -0.0006 |
| **Mean** | **0.0927** | **0.0877** | **-0.0050** |

**AS-0 aggregate (reported):** `Mean JSD = 0.0877 +- 0.0451`, `Mean Pearson r = -0.2084 +- 0.3198`, `Mean Latency = 299.2 ms`.

## LaTeX (Copy Into Paper)

```latex
\begin{table}[t]
\centering
\caption{Aggregate baseline comparison against the main method (SWA-MPPI on Llama-3.1-70B). Lower JSD is better.}
\label{tab:baseline_aggregate}
\begin{tabular}{lcccc}
\toprule
Method & Mean JSD $\downarrow$ & Mean Pearson $r$ $\uparrow$ & $\Delta$JSD vs SWA & Relative gap \\
\midrule
\textbf{SWA-MPPI v3 (Main, Llama-70B)} & \textbf{0.0612} & \textbf{0.3840} & 0.0000 & 0.0\% \\
Vanilla (Llama-70B) & 0.0937 & -0.0260 & +0.0325 & +34.7\% \\
Activation Steering (AS-0, method) & 0.0877 & -0.2084 & +0.0265 & +30.2\% \\
B1: CountryInstruct & 0.0680 & 0.2327 & +0.0068 & +11.1\% \\
B3: PRISM Prompting & 0.0699 & 0.1475 & +0.0087 & +12.5\% \\
B2: WVS Profile Prompting & 0.0737 & 0.1554 & +0.0125 & +16.9\% \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table*}[t]
\centering
\caption{Per-country JSD comparison across baselines, steering method, and SWA-MPPI (Llama-3.1-70B) (best in bold).}
\label{tab:baseline_country_full}
\begin{tabular}{lcccccc}
\toprule
Country & B1 & B2 & B3 & Steering (AS-0) & Vanilla (Llama) & SWA-MPPI \\
\midrule
USA & 0.1694 & 0.1414 & \textbf{0.1074} & 0.1565 & 0.1561 & 0.1234 \\
DEU & 0.0501 & 0.0482 & 0.0556 & 0.0656 & 0.0709 & \textbf{0.0391} \\
CHN & 0.0516 & 0.0356 & 0.0499 & 0.0488 & 0.0459 & \textbf{0.0311} \\
JPN & 0.0442 & 0.0432 & \textbf{0.0376} & 0.0460 & 0.0457 & 0.0416 \\
BRA & 0.0552 & \textbf{0.0430} & 0.0598 & 0.0570 & 0.0532 & 0.0656 \\
SAU & 0.0406 & \textbf{0.0351} & 0.0545 & 0.0596 & 0.0539 & 0.0596 \\
VNM & 0.0557 & 0.0784 & 0.0695 & 0.0892 & 0.1546 & \textbf{0.0529} \\
FRA & \textbf{0.0463} & 0.0470 & 0.0744 & 0.0542 & 0.0500 & 0.0516 \\
IND & 0.0443 & 0.0464 & 0.0499 & 0.0494 & 0.0442 & \textbf{0.0337} \\
KOR & 0.0777 & 0.0715 & 0.0578 & 0.0700 & 0.0893 & \textbf{0.0436} \\
GBR & 0.0967 & 0.1280 & 0.1015 & 0.1589 & 0.1568 & \textbf{0.0918} \\
RUS & 0.0456 & 0.0708 & 0.0537 & 0.0525 & 0.0449 & \textbf{0.0413} \\
MEX & \textbf{0.0384} & 0.0718 & 0.0739 & 0.0836 & 0.1265 & 0.0425 \\
NGA & 0.1006 & 0.1179 & \textbf{0.0988} & 0.1642 & 0.1582 & \textbf{0.0988} \\
AUS & 0.1033 & 0.1275 & 0.1036 & 0.1600 & 0.1551 & \textbf{0.1010} \\
\bottomrule
\end{tabular}
\end{table*}
```

## Repro Notes
- `Vanilla (Llama)` and `SWA (Llama)` are direct pair from the same run block you provided.
- `Activation Steering (AS-0)` is tracked as a method study / ablation, not as a standalone baseline family.
- `B1/B2/B3` are also Llama-based standalone runs, so the aggregate table is backbone-consistent.
- Artifact files: `results/b1_*`, `results/b2_*`, `results/b3_*`, and SWA/vanilla comparison outputs.
