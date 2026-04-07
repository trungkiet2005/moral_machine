#!/usr/bin/env python3
"""
Round3 experiment: Prospect-Theory parameter sensitivity.

Why:
  Reviewer asks whether SWA-MPPI is sensitive to (alpha, beta, kappa).

What this script does:
  - Reuses the production SWA-MPPI pipeline from `swa_mppi_ablation.py`
  - Sweeps PT parameters on a representative country subset
  - Reports mean JSD / Pearson-r per setting

Notes:
  - Standalone runner for local/Kaggle execution
  - Uses the same model + data loading path as existing experiments
"""

import os
import gc
import itertools
import random as _rng
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch

import swa_mppi_ablation as swa


@dataclass
class PTExpConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    n_scenarios: int = 500
    target_countries: List[str] = field(default_factory=lambda: ["USA", "DEU", "JPN", "BRA", "GBR"])
    multitp_data_path: str = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
    multitp_translator: str = "google"
    multitp_suffix: str = ""
    wvs_data_path: str = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    human_amce_path: str = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    output_dir: str = "/kaggle/working/SWA_MPPI/results"

    # Reviewer-targeted PT sweep grid
    alpha_values: List[float] = field(default_factory=lambda: [0.80, 0.88, 0.95])
    beta_values: List[float] = field(default_factory=lambda: [0.80, 0.88, 0.95])
    kappa_values: List[float] = field(default_factory=lambda: [1.80, 2.25, 2.80])


def _build_base_swa_cfg(exp_cfg: PTExpConfig) -> swa.SWAConfig:
    cfg = swa.SWAConfig()
    cfg.model_name = exp_cfg.model_name
    cfg.max_seq_length = exp_cfg.max_seq_length
    cfg.load_in_4bit = exp_cfg.load_in_4bit
    cfg.n_scenarios = exp_cfg.n_scenarios
    cfg.target_countries = list(exp_cfg.target_countries)
    cfg.multitp_data_path = exp_cfg.multitp_data_path
    cfg.multitp_translator = exp_cfg.multitp_translator
    cfg.multitp_suffix = exp_cfg.multitp_suffix
    cfg.wvs_data_path = exp_cfg.wvs_data_path
    cfg.human_amce_path = exp_cfg.human_amce_path
    cfg.output_dir = exp_cfg.output_dir
    return cfg


def main() -> None:
    from unsloth import FastLanguageModel
    from transformers import logging as tlog

    tlog.set_verbosity_error()
    _rng.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    exp_cfg = PTExpConfig()
    os.makedirs(exp_cfg.output_dir, exist_ok=True)

    print("=" * 72)
    print("PT PARAMETER SENSITIVITY (Round3)")
    print(f"Countries: {exp_cfg.target_countries}")
    print(f"Grid size: {len(exp_cfg.alpha_values)} x {len(exp_cfg.beta_values)} x {len(exp_cfg.kappa_values)}")
    print("=" * 72)

    print(f"[MODEL] Loading {exp_cfg.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=exp_cfg.model_name,
        max_seq_length=exp_cfg.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=exp_cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Preload country data/personas once for fairness across PT settings.
    per_country_data = {}
    per_country_personas = {}
    base_cfg = _build_base_swa_cfg(exp_cfg)
    for country in exp_cfg.target_countries:
        lang = swa._COUNTRY_LANG.get(country, "en")
        df = swa.load_multitp_dataset(
            data_base_path=exp_cfg.multitp_data_path,
            lang=lang,
            translator=exp_cfg.multitp_translator,
            suffix=exp_cfg.multitp_suffix,
            n_scenarios=exp_cfg.n_scenarios,
        )
        df = swa.balance_scenario_dataset(df, min_per_category=50, seed=42, lang=lang)
        per_country_data[country] = df
        per_country_personas[country] = swa.build_country_personas(country, wvs_path=exp_cfg.wvs_data_path)

    rows = []
    grid = list(itertools.product(exp_cfg.alpha_values, exp_cfg.beta_values, exp_cfg.kappa_values))
    for i, (alpha, beta, kappa) in enumerate(grid, start=1):
        cfg = _build_base_swa_cfg(exp_cfg)
        cfg.pt_alpha = float(alpha)
        cfg.pt_beta = float(beta)
        cfg.pt_kappa = float(kappa)
        print(f"\n[{i}/{len(grid)}] PT(alpha={alpha:.2f}, beta={beta:.2f}, kappa={kappa:.2f})")

        jsds, rs = [], []
        for country in exp_cfg.target_countries:
            _, summary = swa.run_country_experiment(
                model=model,
                tokenizer=tokenizer,
                country_iso=country,
                personas=per_country_personas[country],
                scenario_df=per_country_data[country],
                cfg=cfg,
            )
            align = summary.get("alignment", {})
            jsd = align.get("jsd", np.nan)
            r = align.get("pearson_r", np.nan)
            jsds.append(jsd)
            rs.append(r)
            rows.append(
                {
                    "country": country,
                    "pt_alpha": alpha,
                    "pt_beta": beta,
                    "pt_kappa": kappa,
                    "jsd": jsd,
                    "pearson_r": r,
                }
            )
            print(f"  {country}: JSD={jsd:.4f}, r={r:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

        print(f"  => mean JSD={np.nanmean(jsds):.4f}, mean r={np.nanmean(rs):.4f}")

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby(["pt_alpha", "pt_beta", "pt_kappa"], as_index=False)
        .agg(mean_jsd=("jsd", "mean"), mean_pearson_r=("pearson_r", "mean"))
        .sort_values(["mean_jsd", "mean_pearson_r"], ascending=[True, False])
        .reset_index(drop=True)
    )

    detail_path = os.path.join(exp_cfg.output_dir, "pt_sensitivity_per_country.csv")
    summary_path = os.path.join(exp_cfg.output_dir, "pt_sensitivity_summary.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("PT sensitivity completed.")
    print(f"[SAVE] {detail_path}")
    print(f"[SAVE] {summary_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(
            f"[BEST] alpha={best['pt_alpha']:.2f}, beta={best['pt_beta']:.2f}, "
            f"kappa={best['pt_kappa']:.2f} | JSD={best['mean_jsd']:.4f}, r={best['mean_pearson_r']:.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()

