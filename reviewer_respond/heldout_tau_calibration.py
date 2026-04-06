#!/usr/bin/env python3
"""
Held-out tau calibration experiment for Round2 response.

This script reuses the verified SWA-MPPI implementation in `swa_mppi.py` and
compares 3 conditions:
  A) tau calibrated on held-out CALIB split (20%), evaluated on EVAL split (80%)
  B) tau calibrated directly on EVAL split (leakage baseline)
  C) fixed tau (no calibration)

Outputs:
  - tau_holdout_summary.csv
  - tau_per_country.csv
"""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swa_mppi import (
    SWAConfig,
    _COUNTRY_LANG,
    ImplicitSWAController,
    balance_scenario_dataset,
    build_country_personas,
    compute_alignment_metrics,
    compute_amce_from_preferences,
    generate_multitp_scenarios,
    load_human_amce,
    load_multitp_dataset,
)


@dataclass
class HeldoutTauConfig:
    # Base SWA config from swa_mppi.py
    base: SWAConfig = field(default_factory=SWAConfig)
    calib_fraction: float = 0.20
    random_seed: int = 42
    fixed_tau: float = 0.001
    output_subdir: str = "tau_holdout"
    target_countries: List[str] = field(
        default_factory=lambda: [
            "USA", "DEU", "CHN", "JPN", "BRA", "SAU", "VNM",
            "GBR", "KOR", "RUS", "MEX", "NGA", "AUS", "FRA", "IND",
        ]
    )


def _build_country_df(cfg: HeldoutTauConfig, country: str) -> pd.DataFrame:
    lang = _COUNTRY_LANG.get(country, "en")
    try:
        df = load_multitp_dataset(
            data_base_path=cfg.base.multitp_data_path,
            lang=lang,
            translator=cfg.base.multitp_translator,
            suffix=cfg.base.multitp_suffix,
            n_scenarios=cfg.base.n_scenarios,
            seed=cfg.random_seed,
        )
    except Exception:
        # Fallback to synthetic if MultiTP file is unavailable
        df = generate_multitp_scenarios(
            n_scenarios=cfg.base.n_scenarios,
            seed=cfg.random_seed,
            lang=lang,
        )

    # Important fix: argument is `min_per_category`, not `min_per`
    return balance_scenario_dataset(
        scenario_df=df,
        min_per_category=50,
        seed=cfg.random_seed,
        lang=lang,
    )


def _split_calib_eval(df: pd.DataFrame, calib_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_calib = max(10, int(len(df) * calib_fraction))
    calib_df = df.iloc[:n_calib].copy()
    eval_df = df.iloc[n_calib:].copy()
    return calib_df, eval_df


@torch.no_grad()
def _evaluate_with_tau(
    controller: ImplicitSWAController,
    eval_df: pd.DataFrame,
    country: str,
    human_amce_path: str,
) -> Dict[str, float]:
    lang = _COUNTRY_LANG.get(country, "en")
    rows = []
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Eval[{country}]"):
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        pred = controller.predict(
            user_query=prompt,
            preferred_on_right=bool(row.get("preferred_on_right", 1)),
            phenomenon_category=row.get("phenomenon_category", "default"),
            lang=lang,
        )
        rows.append(
            {
                "scenario_idx": idx,
                "phenomenon_category": row.get("phenomenon_category", "default"),
                "this_group_name": row.get("this_group_name", "Unknown"),
                "preferred_on_right": int(bool(row.get("preferred_on_right", 1))),
                "n_left": int(row.get("n_left", 1)),
                "n_right": int(row.get("n_right", 1)),
                "p_spare_preferred": float(pred["p_spare_preferred"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    model_amce = compute_amce_from_preferences(pred_df)
    human_amce = load_human_amce(human_amce_path, country)

    align = compute_alignment_metrics(model_amce, human_amce)
    return {
        "jsd": float(align.get("jsd", np.nan)),
        "pearson_r": float(align.get("pearson_r", np.nan)),
    }


def main() -> None:
    from transformers import logging as tlog
    from unsloth import FastLanguageModel

    cfg = HeldoutTauConfig()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    tlog.set_verbosity_error()

    out_dir = os.path.join(cfg.base.output_dir, cfg.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[MODEL] Loading: {cfg.base.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base.model_name,
        max_seq_length=cfg.base.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=cfg.base.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    all_results: Dict[str, List[Dict[str, float]]] = {
        "A_holdout": [],
        "B_leakage": [],
        "C_fixed": [],
    }
    tau_log: List[Dict[str, float]] = []

    for country in cfg.target_countries:
        print(f"\n{'=' * 70}\n[COUNTRY] {country}\n{'=' * 70}")
        full_df = _build_country_df(cfg, country)
        calib_df, eval_df = _split_calib_eval(full_df, cfg.calib_fraction, cfg.random_seed)
        print(f"[SPLIT] total={len(full_df)} calib={len(calib_df)} eval={len(eval_df)}")

        personas = build_country_personas(country, wvs_path=cfg.base.wvs_data_path)
        controller = ImplicitSWAController(
            model=model,
            tokenizer=tokenizer,
            personas=personas,
            lambda_coop=cfg.base.lambda_coop,
            alpha_kl=cfg.base.alpha_kl,
            K_samples=cfg.base.K_samples,
            noise_std=cfg.base.noise_std,
            temperature=cfg.base.temperature,
            tau_conflict=cfg.fixed_tau,
            logit_temperature=cfg.base.logit_temperature,
            category_logit_temperatures=cfg.base.category_logit_temperatures,
            pt_alpha=cfg.base.pt_alpha,
            pt_beta=cfg.base.pt_beta,
            pt_kappa=cfg.base.pt_kappa,
            decision_temperature=cfg.base.decision_temperature,
        )

        tau_a = controller.calibrate_tau(
            calibration_df=calib_df,
            target_trigger_rate=cfg.base.tau_target_trigger_rate,
            n_calib=cfg.base.tau_calibration_n,
            lang=_COUNTRY_LANG.get(country, "en"),
        )
        tau_b = controller.calibrate_tau(
            calibration_df=eval_df,
            target_trigger_rate=cfg.base.tau_target_trigger_rate,
            n_calib=cfg.base.tau_calibration_n,
            lang=_COUNTRY_LANG.get(country, "en"),
        )
        tau_c = cfg.fixed_tau
        tau_log.append({"country": country, "tau_A": tau_a, "tau_B": tau_b, "tau_C": tau_c})

        for cond, tau in [("A_holdout", tau_a), ("B_leakage", tau_b), ("C_fixed", tau_c)]:
            controller.tau_conflict = float(tau)
            m = _evaluate_with_tau(
                controller=controller,
                eval_df=eval_df,
                country=country,
                human_amce_path=cfg.base.human_amce_path,
            )
            m["country"] = country
            m["tau"] = float(tau)
            all_results[cond].append(m)
            print(f"[{cond}] tau={tau:.6f} JSD={m['jsd']:.4f} r={m['pearson_r']:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    summary_rows = []
    labels = {
        "A_holdout": "Correct (held-out calib)",
        "B_leakage": "Leaked (same-dist calib)",
        "C_fixed": "Fixed tau (no calib)",
    }
    for cond, rows in all_results.items():
        jsds = [r["jsd"] for r in rows]
        prs = [r["pearson_r"] for r in rows]
        summary_rows.append(
            {
                "Condition": cond,
                "Mean_JSD": float(np.nanmean(jsds)),
                "Mean_Pearson_r": float(np.nanmean(prs)),
                "Interpretation": labels[cond],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    tau_df = pd.DataFrame(tau_log)
    summary_path = os.path.join(out_dir, "tau_holdout_summary.csv")
    tau_path = os.path.join(out_dir, "tau_per_country.csv")
    summary_df.to_csv(summary_path, index=False)
    tau_df.to_csv(tau_path, index=False)

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))
    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {tau_path}")


if __name__ == "__main__":
    main()
