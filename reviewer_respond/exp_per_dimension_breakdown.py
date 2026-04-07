#!/usr/bin/env python3
"""
STANDALONE — Experiment: Per-Dimension MAE Breakdown
=====================================================
Addresses Round2 Q1: "Are gains dominated by the numerosity (utilitarianism)
dimension or distributed across dimensions?"

Loads existing result CSVs/PKLs from previous runs and computes:
  - Per-dimension MAE: Vanilla vs SWA-MPPI vs all baselines
  - Per-dimension AMCE comparison table
  - WVS attribute → dimension AMCE shift correlation (Q4)

Copy this file into a Kaggle cell and run. Requires prior run outputs in /results/.
"""

import os, gc, warnings, pickle, csv as _csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR    = "/kaggle/working/SWA_MPPI/results"   # baselines + experiments
SWA_RESULTS_DIR = "/kaggle/working/results_swa"          # main swa_mppi.py output
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
OUTPUT_DIR = RESULTS_DIR

DIMENSIONS = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism"]
DIM_PREFERRED = {
    "Species": "Species_Humans", "Gender": "Gender_Female", "Age": "Age_Young",
    "Fitness": "Fitness_Fit", "SocialValue": "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}
DIM_LABELS = {
    "Species": "Species\n(Human)", "Gender": "Gender\n(Female)",
    "Age": "Age\n(Young)", "Fitness": "Fitness\n(Fit)",
    "SocialValue": "Social\nValue", "Utilitarianism": "Util.\n(More)",
}
LABEL_TO_CRITERION = {
    "Species": "Species_Humans", "Gender": "Gender_Female", "Age": "Age_Young",
    "Fitness": "Fitness_Fit", "Social Status": "SocialValue_High",
    "No. Characters": "Utilitarianism_More",
}
COUNTRIES = ["USA","DEU","CHN","JPN","BRA","SAU","VNM","FRA","IND","KOR","GBR","RUS","MEX","NGA","AUS"]

# ── WVS Dimensions ────────────────────────────────────────────────────────────
WVS_DIMS = {
    "gender_equality": ["Q58P","Q59P","Q60P"],
    "religion": ["Q6P"],
    "trust": ["Q43P"],
    "moral_permissiveness": ["Q50","Q52P","Q54P"],
    "work_importance": ["Q5P"],
    "family": ["Q1P"],
    "autonomy": ["Q39P"],
    "meritocracy": ["Q40P"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_human_amce_all():
    """Load human AMCE for all countries → {country: {dim_key: float}}"""
    try:
        df = pd.read_csv(HUMAN_AMCE_PATH)
    except FileNotFoundError:
        print(f"[WARN] AMCE file not found: {HUMAN_AMCE_PATH}")
        return {}
    country_col = "Country" if "Country" in df.columns else "ISO3"
    result = {}
    for country in COUNTRIES:
        cdf = df[df[country_col] == country]
        if cdf.empty:
            continue
        vals = {}
        for _, row in cdf.iterrows():
            label = str(row.get("Label", ""))
            if label in LABEL_TO_CRITERION:
                raw = float(row["Estimates"])
                vals[LABEL_TO_CRITERION[label]] = (1.0 + raw) / 2.0 * 100.0
        result[country] = vals
    print(f"[DATA] Loaded human AMCE for {len(result)} countries")
    return result


def compute_amce_from_results_csv(csv_path):
    """Compute per-dimension AMCE from a results CSV."""
    from sklearn.linear_model import LinearRegression
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {}
    amce = {}
    prob_col = "p_spare_preferred" if "p_spare_preferred" in df.columns else "lp_p_right"
    for dim in DIMENSIONS:
        cat_df = df[df["phenomenon_category"] == dim]
        if len(cat_df) < 3:
            continue
        p_vals = cat_df[prob_col].values.astype(np.float64)
        if dim == "Utilitarianism":
            por = cat_df["preferred_on_right"].values
            n_r = cat_df["n_right"].values.astype(np.float64)
            n_l = cat_df["n_left"].values.astype(np.float64)
            n_diff = np.abs(np.where(por==1, n_r, n_l) - np.where(por==1, n_l, n_r))
            valid = n_diff > 0
            if valid.sum() < 3:
                continue
            reg = LinearRegression(fit_intercept=True)
            reg.fit(n_diff[valid].reshape(-1,1), p_vals[valid])
            amce_val = float(reg.predict([[float(n_diff[valid].mean())]])[0]) * 100.0
        else:
            amce_val = float(p_vals.mean()) * 100.0
        amce[DIM_PREFERRED[dim]] = float(np.clip(amce_val, 0.0, 100.0))
    return amce


def load_all_method_results():
    """Load AMCE for SWA-MPPI (results_swa/), Vanilla and baselines (SWA_MPPI/results/)."""
    # SWA-MPPI saved by main swa_mppi.py → results_swa/
    swa_patterns  = {"SWA-MPPI": "swa_results_{}.csv", "Vanilla": "vanilla_results_{}.csv"}
    base_patterns = {
        "B1-CountryInstruct":  "b1_country_instruct_{}.csv",
        "B2-ProfilePrompt":    "b2_profile_prompt_{}.csv",
        "B3-PRISM":            "b3_prism_{}.csv",
        "B4-PersonaVoting":    "b4_persona_voting_{}.csv",
        "B5-PersonaConsensus": "b5_persona_consensus_{}.csv",
    }
    all_amce = {}
    for method, pattern in swa_patterns.items():
        all_amce[method] = {}
        for country in COUNTRIES:
            for d in [SWA_RESULTS_DIR, RESULTS_DIR]:   # try results_swa first
                csv_path = os.path.join(d, pattern.format(country))
                amce = compute_amce_from_results_csv(csv_path)
                if amce:
                    all_amce[method][country] = amce
                    break
        print(f"[LOAD] {method:<22s}: {len(all_amce[method]):2d} countries  (swa_results_swa/)")
    for method, pattern in base_patterns.items():
        all_amce[method] = {}
        for country in COUNTRIES:
            csv_path = os.path.join(RESULTS_DIR, pattern.format(country))
            amce = compute_amce_from_results_csv(csv_path)
            if amce:
                all_amce[method][country] = amce
        print(f"[LOAD] {method:<22s}: {len(all_amce[method]):2d} countries  (SWA_MPPI/results/)")
    return all_amce


def load_wvs_profiles():
    """Load WVS country-level aggregate profiles."""
    all_vars = set()
    for vars_list in WVS_DIMS.values():
        all_vars.update(vars_list)
    all_vars.update(["Q261", "A_YEAR"])
    data = defaultdict(lambda: defaultdict(list))
    try:
        with open(WVS_PATH, 'r') as f:
            reader = _csv.reader(f)
            header = next(reader)
            cidx = header.index("B_COUNTRY_ALPHA")
            var_idx = {v: header.index(v) for v in all_vars if v in header}
            for row in reader:
                country = row[cidx]
                if country not in COUNTRIES:
                    continue
                try:
                    birth = float(row[var_idx["Q261"]])
                    syear = float(row[var_idx["A_YEAR"]])
                    if birth < 1900 or birth > 2010 or syear < 2015:
                        continue
                except:
                    continue
                for var in all_vars:
                    if var in ("Q261", "A_YEAR"):
                        continue
                    try:
                        val = float(row[var_idx[var]])
                        if val > 0:
                            data[country][var].append(val)
                    except:
                        pass
    except FileNotFoundError:
        print(f"[WARN] WVS file not found: {WVS_PATH}")
        return {}
    profiles = {}
    for c in COUNTRIES:
        profiles[c] = {}
        for dim_name, vars_list in WVS_DIMS.items():
            vals = []
            for v in vars_list:
                vals.extend(data[c][v])
            profiles[c][dim_name] = round(sum(vals)/len(vals), 3) if vals else np.nan
    n_loaded = sum(1 for c in profiles if not np.isnan(profiles[c].get("religion", np.nan)))
    print(f"[WVS] Loaded profiles for {n_loaded}/{len(COUNTRIES)} countries")
    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-Dimension MAE Breakdown (Q1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_per_dimension_mae(all_amce, human_amce):
    """
    For each method × dimension: compute mean |model_amce - human_amce| across countries.
    Returns DataFrame: rows=methods, columns=dimensions + Mean.
    """
    rows = []
    for method, country_amce in all_amce.items():
        row = {"Method": method}
        dim_errors = {dim: [] for dim in DIMENSIONS}
        for country in COUNTRIES:
            if country not in country_amce or country not in human_amce:
                continue
            m_amce = country_amce[country]
            h_amce = human_amce[country]
            for dim in DIMENSIONS:
                key = DIM_PREFERRED[dim]
                if key in m_amce and key in h_amce:
                    dim_errors[dim].append(abs(m_amce[key] - h_amce[key]))
        for dim in DIMENSIONS:
            row[dim] = np.mean(dim_errors[dim]) if dim_errors[dim] else np.nan
        row["Mean"] = np.nanmean([row[d] for d in DIMENSIONS])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def plot_per_dimension_mae(mae_df, output_dir):
    """Figure: heatmap + bar chart of per-dimension MAE."""
    methods_order = [m for m in [
        "Vanilla", "B1-CountryInstruct", "B2-ProfilePrompt", "B3-PRISM",
        "B4-PersonaVoting", "B5-PersonaConsensus", "SWA-MPPI"
    ] if m in mae_df.index]
    dims_order = DIMENSIONS

    data_mat = mae_df.loc[methods_order, dims_order].values

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(data_mat, cmap="YlOrRd", aspect="auto",
                    vmin=0, vmax=np.nanmax(data_mat)*1.1)
    ax1.set_xticks(range(len(dims_order)))
    ax1.set_xticklabels([DIM_LABELS[d] for d in dims_order], fontsize=10)
    ax1.set_yticks(range(len(methods_order)))
    ax1.set_yticklabels(methods_order, fontsize=10)
    ax1.set_title("(a) Per-Dimension MAE (pp) by Method", fontsize=13, fontweight="bold")
    for i in range(len(methods_order)):
        for j in range(len(dims_order)):
            val = data_mat[i, j]
            color = "white" if val > np.nanmax(data_mat)*0.6 else "black"
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=9, color=color, fontweight="bold" if methods_order[i]=="SWA-MPPI" else "normal")
    # Highlight SWA-MPPI row
    swa_idx = methods_order.index("SWA-MPPI") if "SWA-MPPI" in methods_order else -1
    if swa_idx >= 0:
        rect = plt.Rectangle((-0.5, swa_idx-0.5), len(dims_order), 1,
                              fill=False, edgecolor="#2196F3", linewidth=2.5)
        ax1.add_patch(rect)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label("MAE (pp)", fontsize=11)

    # Bar: improvement = Vanilla MAE - SWA MAE per dimension
    ax2 = axes[1]
    if "Vanilla" in mae_df.index and "SWA-MPPI" in mae_df.index:
        improvements = [mae_df.loc["Vanilla", d] - mae_df.loc["SWA-MPPI", d] for d in dims_order]
        colors = ["#2196F3" if v > 0 else "#E53935" for v in improvements]
        bars = ax2.bar(range(len(dims_order)), improvements, color=colors, edgecolor="white", width=0.6)
        ax2.set_xticks(range(len(dims_order)))
        ax2.set_xticklabels([DIM_LABELS[d] for d in dims_order], fontsize=10)
        ax2.set_ylabel("MAE Reduction (pp, +ve = improvement)", fontsize=11)
        ax2.set_title("(b) SWA-MPPI MAE Reduction vs. Vanilla per Dimension", fontsize=13, fontweight="bold")
        ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
        for bar, val in zip(bars, improvements):
            ax2.text(bar.get_x()+bar.get_width()/2, val + 0.2 if val >= 0 else val - 0.5,
                     f"{val:+.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_per_dimension_mae.pdf")
    plt.savefig(path, bbox_inches="tight"); plt.savefig(path.replace(".pdf",".png"))
    plt.show(); plt.close()
    print(f"[FIG] Saved → {path}")


def print_per_dim_table(mae_df):
    """Print LaTeX-ready per-dimension MAE table."""
    methods_order = [m for m in [
        "Vanilla","B1-CountryInstruct","B2-ProfilePrompt","B3-PRISM",
        "B4-PersonaVoting","B5-PersonaConsensus","SWA-MPPI"
    ] if m in mae_df.index]
    cols = DIMENSIONS + ["Mean"]

    print("\n" + "="*90)
    print("  PER-DIMENSION MAE TABLE (pp, averaged across 15 countries)")
    print("="*90)
    header = f"  {'Method':<24s}" + "".join(f"  {c:>13s}" for c in [DIM_LABELS[d].replace('\n',' ') for d in DIMENSIONS]) + f"  {'Mean':>8s}"
    print(header)
    print("  " + "-"*86)
    for method in methods_order:
        row_str = f"  {method:<24s}"
        for dim in DIMENSIONS:
            val = mae_df.loc[method, dim] if method in mae_df.index else np.nan
            row_str += f"  {f'{val:.1f}':>13s}" if not np.isnan(val) else f"  {'—':>13s}"
        mean_val = mae_df.loc[method, "Mean"] if method in mae_df.index else np.nan
        row_str += f"  {f'{mean_val:.1f}':>8s}"
        if method == "SWA-MPPI":
            row_str += "  ◀ OURS"
        print(row_str)
    print("="*90)

    # Check if gains are uniform or dominated by utilitarianism
    if "Vanilla" in mae_df.index and "SWA-MPPI" in mae_df.index:
        improvements = {d: mae_df.loc["Vanilla", d] - mae_df.loc["SWA-MPPI", d] for d in DIMENSIONS}
        print(f"\n  SWA-MPPI improvement over Vanilla per dimension:")
        for d, v in sorted(improvements.items(), key=lambda x: -x[1]):
            bar = "█" * int(abs(v)/2)
            sign = "+" if v > 0 else ""
            print(f"    {d:15s}: {sign}{v:.1f} pp  {bar}")
        util_imp = improvements.get("Utilitarianism", 0)
        other_imp = np.mean([v for d, v in improvements.items() if d != "Utilitarianism"])
        print(f"\n  Utilitarianism improvement: {util_imp:.1f} pp")
        print(f"  Other dimensions (mean):    {other_imp:.1f} pp")
        if util_imp > other_imp * 2:
            print(f"  → Gains DOMINATED by Utilitarianism dimension")
        else:
            print(f"  → Gains BROADLY DISTRIBUTED across dimensions")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: WVS → Dimension Correlation (Q4)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wvs_dimension_correlations(all_amce, human_amce, wvs_profiles):
    """
    For each WVS attribute × trolley dimension:
    Compute Pearson r between WVS country score and AMCE shift (SWA - Vanilla).
    This tests the reviewer's Q4: do WVS attributes correlate with specific dimension shifts?
    """
    # Compute per-country per-dimension AMCE shift
    shifts = {}  # {country: {dim_key: shift}}
    for country in COUNTRIES:
        if country not in all_amce.get("SWA-MPPI", {}) or country not in all_amce.get("Vanilla", {}):
            continue
        swa = all_amce["SWA-MPPI"][country]
        van = all_amce["Vanilla"][country]
        h   = human_amce.get(country, {})
        shifts[country] = {}
        for dim in DIMENSIONS:
            key = DIM_PREFERRED[dim]
            if key in swa and key in van and key in h:
                # Shift toward human: positive = SWA brings model closer to human
                van_err = abs(van[key] - h[key])
                swa_err = abs(swa[key] - h[key])
                shifts[country][dim] = van_err - swa_err   # improvement in pp
    countries_with_data = [c for c in shifts if shifts[c] and wvs_profiles.get(c)]

    # Correlation matrix: WVS_dim × trolley_dim
    wvs_dim_names = list(WVS_DIMS.keys())
    corr_mat = np.full((len(wvs_dim_names), len(DIMENSIONS)), np.nan)
    pval_mat = np.full((len(wvs_dim_names), len(DIMENSIONS)), np.nan)

    for wi, wvs_dim in enumerate(wvs_dim_names):
        for di, troll_dim in enumerate(DIMENSIONS):
            wvs_scores = []
            troll_shifts = []
            for c in countries_with_data:
                ws = wvs_profiles[c].get(wvs_dim, np.nan)
                ts = shifts[c].get(troll_dim, np.nan)
                if not np.isnan(ws) and not np.isnan(ts):
                    wvs_scores.append(ws)
                    troll_shifts.append(ts)
            if len(wvs_scores) >= 5:
                r, p = pearsonr(wvs_scores, troll_shifts)
                corr_mat[wi, di] = r
                pval_mat[wi, di] = p

    return corr_mat, pval_mat, wvs_dim_names, countries_with_data


def plot_wvs_correlation(corr_mat, pval_mat, wvs_dim_names, output_dir):
    """Heatmap: WVS attribute × trolley dimension Pearson r."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSIONS], fontsize=10)
    ax.set_yticks(range(len(wvs_dim_names)))
    ax.set_yticklabels([w.replace("_", " ").title() for w in wvs_dim_names], fontsize=10)
    ax.set_title("WVS Attribute × Trolley Dimension: Pearson r\n(SWA-MPPI improvement vs. Vanilla per country)",
                 fontsize=12, fontweight="bold")
    for i in range(len(wvs_dim_names)):
        for j in range(len(DIMENSIONS)):
            val = corr_mat[i, j]
            pval = pval_mat[i, j]
            if not np.isnan(val):
                sig = "*" if pval < 0.05 else ""
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                        fontsize=8.5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=11)
    ax.set_xlabel("Trolley Dimension", fontsize=11)
    ax.set_ylabel("WVS Attribute", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_wvs_dimension_correlation.pdf")
    plt.savefig(path, bbox_inches="tight"); plt.savefig(path.replace(".pdf",".png"))
    plt.show(); plt.close()
    print(f"[FIG] Saved → {path}")

    # Print significant correlations
    print("\n  Significant WVS → Trolley correlations (p < 0.05):")
    for i, wdim in enumerate(wvs_dim_names):
        for j, tdim in enumerate(DIMENSIONS):
            if not np.isnan(corr_mat[i,j]) and pval_mat[i,j] < 0.05:
                print(f"    {wdim:25s} ↔ {tdim:15s}: r={corr_mat[i,j]:+.3f} (p={pval_mat[i,j]:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def save_latex_per_dim_table(mae_df, output_dir):
    methods_order = [m for m in [
        "Vanilla","B1-CountryInstruct","B2-ProfilePrompt","B3-PRISM",
        "B4-PersonaVoting","B5-PersonaConsensus","SWA-MPPI"
    ] if m in mae_df.index]
    dim_short = ["Species","Gender","Age","Fitness","Social","Util.","Mean"]
    path = os.path.join(output_dir, "table_per_dimension_mae.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Per-dimension MAE (pp) averaged over 15 countries. "
                "Lower is better. * p<0.05 improvement over Vanilla.}\n")
        f.write("\\label{tab:per_dim_mae}\n\\small\n")
        f.write("\\begin{tabular}{l" + "c"*len(dim_short) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(dim_short) + " \\\\\n\\midrule\n")
        for method in methods_order:
            row = method
            for dim in DIMENSIONS:
                val = mae_df.loc[method, dim] if method in mae_df.index else np.nan
                row += f" & {val:.1f}" if not np.isnan(val) else " & —"
            mean_val = mae_df.loc[method, "Mean"] if method in mae_df.index else np.nan
            row += f" & {mean_val:.1f}" if not np.isnan(mean_val) else " & —"
            if method == "SWA-MPPI":
                row = "\\textbf{" + row + "}"
            f.write(row + " \\\\\n")
            if method == "B5-PersonaConsensus":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[LaTeX] Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family":"serif","font.size":11,"figure.dpi":150,
                         "savefig.dpi":300,"savefig.bbox":"tight",
                         "axes.grid":True,"grid.alpha":0.3})

    print("=" * 60)
    print("  EXPERIMENT: Per-Dimension MAE Breakdown (Round2 Q1 + Q4)")
    print("=" * 60)

    # Load data
    human_amce   = load_human_amce_all()
    all_amce     = load_all_method_results()
    wvs_profiles = load_wvs_profiles()

    if not human_amce:
        print("[ERROR] Human AMCE data not found. Cannot proceed.")
        return
    if not any(all_amce[m] for m in all_amce):
        print("[ERROR] No method result CSVs found in", RESULTS_DIR)
        print("        Run main.py + baselines first, then re-run this analysis.")
        return

    # ── Analysis 1: Per-dimension MAE table ────────────────────────────────
    print("\n[1/3] Computing per-dimension MAE table...")
    mae_df = compute_per_dimension_mae(all_amce, human_amce)
    print_per_dim_table(mae_df)
    plot_per_dimension_mae(mae_df, OUTPUT_DIR)
    save_latex_per_dim_table(mae_df, OUTPUT_DIR)
    mae_df.to_csv(os.path.join(OUTPUT_DIR, "per_dimension_mae.csv"))
    print(f"[SAVE] per_dimension_mae.csv")

    # ── Analysis 2: WVS → Dimension correlations (Q4) ──────────────────────
    print("\n[2/3] Computing WVS attribute → trolley dimension correlations...")
    if wvs_profiles and all_amce.get("SWA-MPPI") and all_amce.get("Vanilla"):
        corr_mat, pval_mat, wvs_dim_names, countries_used = \
            compute_wvs_dimension_correlations(all_amce, human_amce, wvs_profiles)
        print(f"  Using {len(countries_used)} countries with both WVS and result data")
        plot_wvs_correlation(corr_mat, pval_mat, wvs_dim_names, OUTPUT_DIR)
        # Save correlation matrix
        corr_df = pd.DataFrame(corr_mat, index=wvs_dim_names, columns=DIMENSIONS)
        corr_df.to_csv(os.path.join(OUTPUT_DIR, "wvs_dimension_correlation.csv"))
        pval_df = pd.DataFrame(pval_mat, index=wvs_dim_names, columns=DIMENSIONS)
        pval_df.to_csv(os.path.join(OUTPUT_DIR, "wvs_dimension_pvalues.csv"))
        print(f"[SAVE] wvs_dimension_correlation.csv, wvs_dimension_pvalues.csv")
    else:
        print("  [SKIP] WVS data or SWA/Vanilla results not available.")

    # ── Analysis 3: Summary stats ───────────────────────────────────────────
    print("\n[3/3] Summary statistics...")
    if "SWA-MPPI" in mae_df.index and "Vanilla" in mae_df.index:
        print(f"\n  Overall MAE (mean across all 6 dimensions × 15 countries):")
        print(f"  {'Method':<24s} {'MAE':>8s}")
        print(f"  {'-'*34}")
        methods_order = [m for m in [
            "Vanilla","B1-CountryInstruct","B2-ProfilePrompt","B3-PRISM",
            "B4-PersonaVoting","B5-PersonaConsensus","SWA-MPPI"
        ] if m in mae_df.index]
        for m in methods_order:
            mean_val = mae_df.loc[m, "Mean"]
            marker = " ◀ OURS" if m == "SWA-MPPI" else ""
            print(f"  {m:<24s} {mean_val:>8.2f} pp{marker}")

    print(f"\n[DONE] All outputs saved to {OUTPUT_DIR}/")
    print("  Key files:")
    print("  - fig_per_dimension_mae.pdf    (Figure for paper)")
    print("  - fig_wvs_dimension_correlation.pdf  (Q4 correlation)")
    print("  - table_per_dimension_mae.tex  (LaTeX table for paper)")
    print("  - per_dimension_mae.csv")


# Run directly in Kaggle notebook cell
main()
