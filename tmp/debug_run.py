#!/usr/bin/env python3
"""
Quick debug script: load model once, then interactively trace SWA-MPPI
step-by-step on individual scenarios using debug_predict().

Usage (Kaggle / local):
    python debug_run.py                     # interactive mode
    python debug_run.py --country VNM       # run 3 sample scenarios for VNM
    python debug_run.py --country USA --n 5 # run 5 sample scenarios for USA
    python debug_run.py --prompt "An autonomous vehicle ..."  # custom prompt
"""

import argparse, os, sys, random, warnings
import torch
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import everything from main ──
from main import (
    SWAConfig,
    ImplicitSWAController,
    build_country_personas,
    generate_multitp_scenarios,
    balance_scenario_dataset,
    _COUNTRY_LANG,
    _PROMPT_FRAME_I18N,
)


def load_model(cfg: SWAConfig):
    """Load model + tokenizer (same as main())."""
    import transformers
    transformers.logging.set_verbosity_error()
    from unsloth import FastLanguageModel

    print(f"[MODEL] Loading {cfg.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
    return model, tokenizer


def build_controller(model, tokenizer, country: str, cfg: SWAConfig):
    """Build SWA controller for a given country."""
    lang = _COUNTRY_LANG.get(country, "en")
    personas = build_country_personas(country, wvs_path=cfg.wvs_data_path)

    print(f"[CTRL] Building controller for {country} (lang={lang})")
    print(f"  Personas ({len(personas)}):")
    for i, p in enumerate(personas):
        print(f"    [{i}] {p[:100]}{'...' if len(p) > 100 else ''}")

    controller = ImplicitSWAController(
        model, tokenizer, personas,
        lambda_coop=cfg.lambda_coop,
        alpha_kl=cfg.alpha_kl,
        K_samples=cfg.K_samples,
        noise_std=cfg.noise_std,
        temperature=cfg.temperature,
        tau_conflict=cfg.tau_conflict,
        logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=cfg.category_logit_temperatures,
        pt_alpha=cfg.pt_alpha,
        pt_beta=cfg.pt_beta,
        pt_kappa=cfg.pt_kappa,
        decision_temperature=cfg.decision_temperature,
    )
    return controller, lang


def get_sample_scenarios(lang: str, n: int = 3, seed: int = 42):
    """Generate a few synthetic scenarios for debugging."""
    df = generate_multitp_scenarios(n_scenarios=max(n * 6, 60), seed=seed, lang=lang)
    df = balance_scenario_dataset(df, min_per_category=1, seed=seed, lang=lang)
    # Pick one per category, then fill up to n
    categories = df["phenomenon_category"].unique()
    picked = []
    for cat in categories:
        sub = df[df["phenomenon_category"] == cat]
        if len(sub) > 0 and len(picked) < n:
            picked.append(sub.iloc[0])
    # Fill remaining
    remaining = df[~df.index.isin([r.name for r in picked])]
    for _, row in remaining.iterrows():
        if len(picked) >= n:
            break
        picked.append(row)
    return picked


def run_single(controller, prompt, preferred_on_right, category, lang):
    """Run debug_predict on a single scenario."""
    result = controller.debug_predict(
        user_query=prompt,
        preferred_on_right=bool(preferred_on_right),
        phenomenon_category=category,
        lang=lang,
    )
    return result


def run_batch(controller, scenarios, lang):
    """Run debug_predict on a list of scenario rows."""
    results = []
    for i, row in enumerate(scenarios):
        print(f"\n{'#' * 72}")
        print(f"  SCENARIO {i + 1} / {len(scenarios)}")
        print(f"  Category: {row['phenomenon_category']}")
        print(f"  Preferred on right: {bool(row['preferred_on_right'])}")
        print(f"{'#' * 72}")
        print(f"\n  Prompt: {row['Prompt'][:300]}{'...' if len(row['Prompt']) > 300 else ''}\n")

        result = run_single(
            controller,
            prompt=row["Prompt"],
            preferred_on_right=row["preferred_on_right"],
            category=row["phenomenon_category"],
            lang=lang,
        )
        results.append(result)
    return results


def interactive_mode(controller, lang):
    """Interactive REPL: type a scenario or 'q' to quit."""
    print("\n" + "=" * 72)
    print("  INTERACTIVE DEBUG MODE")
    print("  Type a scenario prompt, or one of:")
    print("    sample [category]  — generate & run a sample (e.g. 'sample Age')")
    print("    tau <value>        — change tau_conflict")
    print("    temp <value>       — change decision_temperature")
    print("    q                  — quit")
    print("=" * 72)

    while True:
        try:
            user_input = input("\n[debug] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() == "q":
            print("Bye!")
            break

        # Command: sample
        if user_input.lower().startswith("sample"):
            parts = user_input.split(maxsplit=1)
            cat_filter = parts[1] if len(parts) > 1 else None
            scenarios = get_sample_scenarios(lang, n=10, seed=random.randint(0, 9999))
            if cat_filter:
                scenarios = [s for s in scenarios
                             if s["phenomenon_category"].lower() == cat_filter.lower()]
            if not scenarios:
                print(f"  No scenarios found for category '{cat_filter}'")
                continue
            row = scenarios[0]
            print(f"  [auto] Category={row['phenomenon_category']}, "
                  f"preferred_right={bool(row['preferred_on_right'])}")
            print(f"  [auto] Prompt: {row['Prompt'][:200]}...")
            run_single(controller, row["Prompt"], row["preferred_on_right"],
                        row["phenomenon_category"], lang)
            continue

        # Command: tau
        if user_input.lower().startswith("tau "):
            try:
                val = float(user_input.split()[1])
                controller.tau_conflict = val
                print(f"  tau_conflict = {val}")
            except (IndexError, ValueError):
                print("  Usage: tau <float>")
            continue

        # Command: temp
        if user_input.lower().startswith("temp "):
            try:
                val = float(user_input.split()[1])
                controller.decision_temperature = val
                print(f"  decision_temperature = {val}")
            except (IndexError, ValueError):
                print("  Usage: temp <float>")
            continue

        # Otherwise: treat as a raw prompt
        cat = input("  Category [Age/Gender/Species/Fitness/SocialValue/Utilitarianism]: ").strip()
        cat = cat if cat else "default"
        pref = input("  Preferred on right? [y/n, default=y]: ").strip().lower()
        preferred_on_right = pref != "n"
        run_single(controller, user_input, preferred_on_right, cat, lang)


def main():
    parser = argparse.ArgumentParser(description="SWA-MPPI Debug Runner")
    parser.add_argument("--country", type=str, default="VNM",
                        help="Country ISO code (default: VNM)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of sample scenarios to run (default: 3)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (skip sample generation)")
    parser.add_argument("--category", type=str, default="Age",
                        help="Category for custom prompt (default: Age)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Enter interactive debug REPL after loading")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run tau calibration before debug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = SWAConfig()
    model, tokenizer = load_model(cfg)
    controller, lang = build_controller(model, tokenizer, args.country, cfg)

    # Optional: calibrate tau
    if args.calibrate:
        print("\n[CALIBRATE] Running tau calibration...")
        calib_df = generate_multitp_scenarios(
            n_scenarios=cfg.tau_calibration_n * 6, seed=args.seed, lang=lang
        )
        controller.calibrate_tau(
            calibration_df=calib_df,
            target_trigger_rate=cfg.tau_target_trigger_rate,
            n_calib=cfg.tau_calibration_n,
            lang=lang,
        )

    if args.interactive:
        interactive_mode(controller, lang)
    elif args.prompt:
        # Single custom prompt
        run_single(controller, args.prompt, True, args.category, lang)
    else:
        # Auto: generate n sample scenarios and run debug on each
        print(f"\n[AUTO] Generating {args.n} sample scenarios for {args.country} (lang={lang})...")
        scenarios = get_sample_scenarios(lang, n=args.n, seed=args.seed)
        run_batch(controller, scenarios, lang)

    print("\n[DONE] Debug session complete.")


if __name__ == "__main__":
    main()
