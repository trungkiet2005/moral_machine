#!/usr/bin/env python3
"""
Standalone analysis: Per-dimension MAE breakdown (Round2 Q1 + Q4).

Fixes over prior version:
1) Robust file discovery: does not hardcode one filename pattern only.
2) Supports nested result folders under RESULTS_DIR.
3) Clear diagnostics for missing methods/countries.
"""

from __future__ import annotations

import csv as _csv
import glob
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
RESULTS_DIR = "/kaggle/working/SWA_MPPI/results"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
OUTPUT_DIR = RESULTS_DIR

COUNTRIES = [
    "USA", "DEU", "CHN", "JPN", "BRA", "SAU", "VNM",
    "FRA", "IND", "KOR", "GBR", "RUS", "MEX", "NGA", "AUS",
]
DIMENSIONS = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism"]
DIM_PREFERRED = {
    "Species": "Species_Humans",
    "Gender": "Gender_Female",
    "Age": "Age_Young",
    "Fitness": "Fitness_Fit",
    "SocialValue": "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}
DIM_LABELS = {
    "Species": "Species\n(Human)",
    "Gender": "Gender\n(Female)",
    "Age": "Age\n(Young)",
    "Fitness": "Fitness\n(Fit)",
    "SocialValue": "Social\nValue",
    "Utilitarianism": "Util.\n(More)",
}
LABEL_TO_CRITERION = {
    "Species": "Species_Humans",
    "Gender": "Gender_Female",
    "Age": "Age_Young",
    "Fitness": "Fitness_Fit",
    "Social Status": "SocialValue_High",
    "No. Characters": "Utilitarianism_More",
}
WVS_DIMS = {
    "gender_equality": ["Q58P", "Q59P", "Q60P"],
    "religion": ["Q6P"],
    "trust": ["Q43P"],
    "moral_permissiveness": ["Q50", "Q52P", "Q54P"],
    "work_importance": ["Q5P"],
    "family": ["Q1P"],
    "autonomy": ["Q39P"],
    "meritocracy": ["Q40P"],
}


def load_human_amce_all() -> Dict[str, Dict[str, float]]:
    try:
        df = pd.read_csv(HUMAN_AMCE_PATH)
    except FileNotFoundError:
        print(f"[WARN] Human AMCE not found: {HUMAN_AMCE_PATH}")
        return {}

    country_col = "Country" if "Country" in df.columns else "ISO3"
    out: Dict[str, Dict[str, float]] = {}
    for country in COUNTRIES:
        cdf = df[df[country_col] == country]
        if cdf.empty:
            continue
        vals: Dict[str, float] = {}
        for _, row in cdf.iterrows():
            label = str(row.get("Label", ""))
            if label not in LABEL_TO_CRITERION:
                continue
            raw = float(row["Estimates"])
            vals[LABEL_TO_CRITERION[label]] = (1.0 + raw) / 2.0 * 100.0
        if vals:
            out[country] = vals
    print(f"[DATA] Human AMCE countries: {len(out)}")
    return out


def compute_amce_from_results_csv(csv_path: str) -> Dict[str, float]:
    from sklearn.linear_model import LinearRegression

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "phenomenon_category" not in df.columns:
        return {}

    prob_col = "p_spare_preferred" if "p_spare_preferred" in df.columns else "lp_p_right"
    if prob_col not in df.columns:
        return {}

    amce: Dict[str, float] = {}
    for dim in DIMENSIONS:
        cat_df = df[df["phenomenon_category"] == dim]
        if len(cat_df) < 3:
            continue

        p_vals = cat_df[prob_col].astype(float).values
        if dim == "Utilitarianism":
            if not {"preferred_on_right", "n_left", "n_right"}.issubset(cat_df.columns):
                continue
            por = cat_df["preferred_on_right"].values
            n_r = cat_df["n_right"].astype(float).values
            n_l = cat_df["n_left"].astype(float).values
            n_diff = np.abs(np.where(por == 1, n_r, n_l) - np.where(por == 1, n_l, n_r))
            valid = n_diff > 0
            if valid.sum() < 3:
                continue
            reg = LinearRegression(fit_intercept=True)
            reg.fit(n_diff[valid].reshape(-1, 1), p_vals[valid])
            amce_val = float(reg.predict([[float(n_diff[valid].mean())]])[0]) * 100.0
        else:
            amce_val = float(np.mean(p_vals)) * 100.0
        amce[DIM_PREFERRED[dim]] = float(np.clip(amce_val, 0.0, 100.0))
    return amce


def _recursive_csvs(root: str) -> List[str]:
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


def _find_country_file(candidates: List[str], country: str, required_tokens: List[str]) -> Optional[str]:
    country_tag = f"_{country}.csv"
    ranked = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        if not bn.endswith(country_tag.lower()):
            continue
        if all(tok in bn for tok in required_tokens):
            ranked.append(p)
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x.count(os.sep), len(x)))
    return ranked[0]


def _method_patterns() -> Dict[str, List[List[str]]]:
    return {
        "SWA-MPPI": [["swa", "result"], ["swa_results"], ["swa-mppi"]],
        "Vanilla": [["vanilla", "result"]],
        "B1-CountryInstruct": [["b1", "country", "instruct"]],
        "B2-ProfilePrompt": [["b2", "profile", "prompt"]],
        "B3-PRISM": [["b3", "prism"]],
        "B4-PersonaVoting": [["b4", "persona", "voting"]],
        "B5-PersonaConsensus": [["b5", "persona", "consensus"]],
    }


def load_all_method_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    all_csvs = _recursive_csvs(RESULTS_DIR)
    if not all_csvs:
        print(f"[WARN] No CSV files found under: {RESULTS_DIR}")
    method_amce: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in _method_patterns()}
    method_files_used: Dict[str, int] = {m: 0 for m in _method_patterns()}

    for method, pattern_options in _method_patterns().items():
        for country in COUNTRIES:
            found_path = None
            for tokens in pattern_options:
                found_path = _find_country_file(all_csvs, country, [t.lower() for t in tokens])
                if found_path:
                    break
            if not found_path:
                continue

            amce = compute_amce_from_results_csv(found_path)
            if amce:
                method_amce[method][country] = amce
                method_files_used[method] += 1

        print(f"[LOAD] {method:<22s}: {len(method_amce[method]):2d} countries "
              f"(files matched: {method_files_used[method]})")
    return method_amce


def load_wvs_profiles() -> Dict[str, Dict[str, float]]:
    all_vars = set()
    for vlist in WVS_DIMS.values():
        all_vars.update(vlist)
    all_vars.update(["Q261", "A_YEAR"])

    data = defaultdict(lambda: defaultdict(list))
    try:
        with open(WVS_PATH, "r", encoding="utf-8") as f:
            rd = _csv.reader(f)
            hdr = next(rd)
            cidx = hdr.index("B_COUNTRY_ALPHA")
            vidx = {v: hdr.index(v) for v in all_vars if v in hdr}

            for row in rd:
                country = row[cidx]
                if country not in COUNTRIES:
                    continue
                try:
                    birth = float(row[vidx["Q261"]])
                    syear = float(row[vidx["A_YEAR"]])
                    if birth < 1900 or birth > 2010 or syear < 2015:
                        continue
                except Exception:
                    continue

                for v in all_vars:
                    if v in ("Q261", "A_YEAR"):
                        continue
                    if v not in vidx:
                        continue
                    try:
                        val = float(row[vidx[v]])
                        if val > 0:
                            data[country][v].append(val)
                    except Exception:
                        pass
    except FileNotFoundError:
        print(f"[WARN] WVS file not found: {WVS_PATH}")
        return {}

    profiles: Dict[str, Dict[str, float]] = {}
    for country in COUNTRIES:
        profiles[country] = {}
        for dim_name, vars_list in WVS_DIMS.items():
            vals: List[float] = []
            for v in vars_list:
                vals.extend(data[country][v])
            profiles[country][dim_name] = float(np.mean(vals)) if vals else np.nan

    n_loaded = sum(1 for c in COUNTRIES if not np.isnan(profiles[c].get("religion", np.nan)))
    print(f"[WVS] Loaded profiles for {n_loaded}/{len(COUNTRIES)} countries")
    return profiles


def compute_per_dimension_mae(
    all_amce: Dict[str, Dict[str, Dict[str, float]]],
    human_amce: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for method, country_amce in all_amce.items():
        row = {"Method": method}
        dim_errors = {dim: [] for dim in DIMENSIONS}
        for country in COUNTRIES:
            if country not in country_amce or country not in human_amce:
                continue
            mvals = country_amce[country]
            hvals = human_amce[country]
            for dim in DIMENSIONS:
                key = DIM_PREFERRED[dim]
                if key in mvals and key in hvals:
                    dim_errors[dim].append(abs(mvals[key] - hvals[key]))

        for dim in DIMENSIONS:
            row[dim] = float(np.mean(dim_errors[dim])) if dim_errors[dim] else np.nan
        row["Mean"] = float(np.nanmean([row[d] for d in DIMENSIONS]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def plot_per_dimension_mae(mae_df: pd.DataFrame, output_dir: str) -> None:
    methods_order = [m for m in [
        "Vanilla", "B1-CountryInstruct", "B2-ProfilePrompt", "B3-PRISM",
        "B4-PersonaVoting", "B5-PersonaConsensus", "SWA-MPPI",
    ] if m in mae_df.index]
    if not methods_order:
        print("[SKIP] No methods to plot.")
        return

    data_mat = mae_df.loc[methods_order, DIMENSIONS].values
    vmax = np.nanmax(data_mat) * 1.1 if np.isfinite(np.nanmax(data_mat)) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    ax1 = axes[0]
    im = ax1.imshow(data_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax1.set_xticks(range(len(DIMENSIONS)))
    ax1.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=10)
    ax1.set_yticks(range(len(methods_order)))
    ax1.set_yticklabels(methods_order, fontsize=10)
    ax1.set_title("(a) Per-Dimension MAE (pp) by Method", fontsize=13, fontweight="bold")

    for i in range(len(methods_order)):
        for j in range(len(DIMENSIONS)):
            val = data_mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > (0.6 * vmax) else "black"
            ax1.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if methods_order[i] == "SWA-MPPI" else "normal",
            )

    if "SWA-MPPI" in methods_order:
        sidx = methods_order.index("SWA-MPPI")
        rect = plt.Rectangle((-0.5, sidx - 0.5), len(DIMENSIONS), 1, fill=False,
                             edgecolor="#2196F3", linewidth=2.5)
        ax1.add_patch(rect)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label("MAE (pp)", fontsize=11)

    ax2 = axes[1]
    if "Vanilla" in mae_df.index and "SWA-MPPI" in mae_df.index:
        improvements = [mae_df.loc["Vanilla", d] - mae_df.loc["SWA-MPPI", d] for d in DIMENSIONS]
        bars = ax2.bar(
            range(len(DIMENSIONS)),
            improvements,
            color=["#2196F3" if v > 0 else "#E53935" for v in improvements],
            edgecolor="white",
            width=0.6,
        )
        ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax2.set_xticks(range(len(DIMENSIONS)))
        ax2.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=10)
        ax2.set_ylabel("MAE Reduction (pp, +ve = improvement)", fontsize=11)
        ax2.set_title("(b) SWA-MPPI MAE Reduction vs Vanilla", fontsize=13, fontweight="bold")
        for bar, val in zip(bars, improvements):
            y = val + 0.2 if val >= 0 else val - 0.5
            ax2.text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.1f}",
                     ha="center", fontsize=9, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Need both Vanilla and SWA-MPPI",
                 transform=ax2.transAxes, ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_per_dimension_mae.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"[FIG] Saved -> {path}")


def compute_wvs_dimension_correlations(
    all_amce: Dict[str, Dict[str, Dict[str, float]]],
    human_amce: Dict[str, Dict[str, float]],
    wvs_profiles: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    shifts: Dict[str, Dict[str, float]] = {}
    for c in COUNTRIES:
        if c not in all_amce.get("SWA-MPPI", {}) or c not in all_amce.get("Vanilla", {}):
            continue
        swa = all_amce["SWA-MPPI"][c]
        van = all_amce["Vanilla"][c]
        hum = human_amce.get(c, {})
        shifts[c] = {}
        for dim in DIMENSIONS:
            key = DIM_PREFERRED[dim]
            if key in swa and key in van and key in hum:
                shifts[c][dim] = abs(van[key] - hum[key]) - abs(swa[key] - hum[key])

    countries_used = [c for c in COUNTRIES if c in shifts and wvs_profiles.get(c)]
    wvs_dim_names = list(WVS_DIMS.keys())
    corr_mat = np.full((len(wvs_dim_names), len(DIMENSIONS)), np.nan)
    pval_mat = np.full((len(wvs_dim_names), len(DIMENSIONS)), np.nan)

    for i, wdim in enumerate(wvs_dim_names):
        for j, tdim in enumerate(DIMENSIONS):
            xs, ys = [], []
            for c in countries_used:
                w = wvs_profiles[c].get(wdim, np.nan)
                y = shifts[c].get(tdim, np.nan)
                if not np.isnan(w) and not np.isnan(y):
                    xs.append(w)
                    ys.append(y)
            if len(xs) >= 5:
                r, p = pearsonr(xs, ys)
                corr_mat[i, j] = r
                pval_mat[i, j] = p

    return corr_mat, pval_mat, wvs_dim_names, countries_used


def plot_wvs_correlation(corr_mat: np.ndarray, pval_mat: np.ndarray, wvs_dim_names: List[str], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=10)
    ax.set_yticks(range(len(wvs_dim_names)))
    ax.set_yticklabels([w.replace("_", " ").title() for w in wvs_dim_names], fontsize=10)
    ax.set_title("WVS Attribute x Trolley Dimension: Pearson r", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trolley Dimension", fontsize=11)
    ax.set_ylabel("WVS Attribute", fontsize=11)

    for i in range(len(wvs_dim_names)):
        for j in range(len(DIMENSIONS)):
            val = corr_mat[i, j]
            if np.isnan(val):
                continue
            p = pval_mat[i, j]
            sig = "*" if p < 0.05 else ""
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=8.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_wvs_dimension_correlation.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"[FIG] Saved -> {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    print("=" * 64)
    print("  EXPERIMENT: Per-Dimension MAE Breakdown (Round2 Q1 + Q4)")
    print("=" * 64)

    human_amce = load_human_amce_all()
    all_amce = load_all_method_results()
    wvs_profiles = load_wvs_profiles()

    if not human_amce:
        print("[ERROR] Missing human AMCE data.")
        return

    if not any(len(v) > 0 for v in all_amce.values()):
        print(f"[ERROR] No result CSV matched under: {RESULTS_DIR}")
        print("        Confirm baseline/main outputs exist and filenames include _{ISO3}.csv")
        return

    print("\n[1/3] Per-dimension MAE")
    mae_df = compute_per_dimension_mae(all_amce, human_amce)
    mae_df.to_csv(os.path.join(OUTPUT_DIR, "per_dimension_mae.csv"))
    print(mae_df.round(3))
    plot_per_dimension_mae(mae_df, OUTPUT_DIR)

    print("\n[2/3] WVS -> dimension correlation")
    if wvs_profiles and all_amce.get("SWA-MPPI") and all_amce.get("Vanilla"):
        corr_mat, pval_mat, wvs_dim_names, countries_used = compute_wvs_dimension_correlations(
            all_amce, human_amce, wvs_profiles
        )
        print(f"  Countries used: {len(countries_used)}")
        plot_wvs_correlation(corr_mat, pval_mat, wvs_dim_names, OUTPUT_DIR)
        pd.DataFrame(corr_mat, index=wvs_dim_names, columns=DIMENSIONS).to_csv(
            os.path.join(OUTPUT_DIR, "wvs_dimension_correlation.csv")
        )
        pd.DataFrame(pval_mat, index=wvs_dim_names, columns=DIMENSIONS).to_csv(
            os.path.join(OUTPUT_DIR, "wvs_dimension_pvalues.csv")
        )
    else:
        print("  [SKIP] Need WVS + both SWA and Vanilla results.")

    print("\n[3/3] Summary")
    if "SWA-MPPI" in mae_df.index and "Vanilla" in mae_df.index:
        imp = mae_df.loc["Vanilla", "Mean"] - mae_df.loc["SWA-MPPI", "Mean"]
        print(f"  Mean MAE reduction (SWA vs Vanilla): {imp:+.3f} pp")
    print(f"\n[DONE] Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
