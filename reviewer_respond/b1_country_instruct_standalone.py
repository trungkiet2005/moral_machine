#!/usr/bin/env python3
"""
Standalone Baseline-1 (Country Instruction Prompt) runner.

Fixes:
- Avoids NameError for SWAConfig by importing required objects from swa_mppi.py.
- Loads model if not already loaded in memory.
"""

from __future__ import annotations

import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from swa_mppi import (
    SWAConfig,
    ChatTemplateHelper,
    _COUNTRY_FULL_NAMES,
    _COUNTRY_LANG,
    _PROMPT_FRAME_I18N,
    _SCENARIO_FRAME_I18N,
    balance_scenario_dataset,
    compute_alignment_metrics,
    compute_amce_from_preferences,
    generate_multitp_scenarios,
    load_human_amce,
    load_multitp_dataset,
)


_B1_TEMPLATES = {
    "en": (
        "You are an average citizen from {country_name}. "
        "Answer the following moral dilemma as someone shaped by the cultural, "
        "religious, and social values typically held in {country_name}. "
        "Your response should reflect common moral intuitions in {country_name}."
    ),
    "zh": "你是一位来自{country_name}的普通公民。请根据该国文化、宗教和社会价值观回答以下道德困境。",
    "ja": "あなたは{country_name}の一般市民です。文化的・宗教的・社会的価値観に基づいて答えてください。",
    "ko": "{country_name}의 평범한 시민으로서 문화/종교/사회적 가치관에 따라 답하세요.",
    "de": "Sie sind ein durchschnittlicher Bürger aus {country_name}. Antworten Sie entsprechend der dortigen Werte.",
    "fr": "Vous êtes un citoyen ordinaire de {country_name}. Répondez selon les valeurs de ce pays.",
    "pt": "Você é um cidadão médio do {country_name}. Responda segundo os valores típicos do país.",
    "ar": "أنت مواطن عادي من {country_name}. أجب وفق القيم السائدة في البلد.",
    "vi": "Bạn là công dân bình thường từ {country_name}. Hãy trả lời theo giá trị đạo đức phổ biến của quốc gia đó.",
    "hi": "आप {country_name} के एक औसत नागरिक हैं। उसी सांस्कृतिक/सामाजिक मूल्यों के आधार पर उत्तर दें।",
    "ru": "Вы рядовой гражданин {country_name}. Отвечайте в соответствии с распространенными ценностями страны.",
    "es": "Usted es un ciudadano promedio de {country_name}. Responda según los valores típicos del país.",
}


def _b1_logit_p_spare(model, full_ids, left_id, right_id, pref_right, decision_temperature=1.0):
    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)
        logits = out.logits[0, -1, :]
        pair = torch.stack([logits[left_id], logits[right_id]])
        probs = F.softmax(pair / decision_temperature, dim=-1)
        p_l, p_r = probs[0].item(), probs[1].item()
    return p_r if pref_right else p_l


def _b1_two_pass(
    model,
    tokenizer,
    chat_helper,
    sys_prompt,
    scenario_text,
    pref_right,
    lang,
    left_id,
    right_id,
    decision_temperature,
    device,
):
    frame = _PROMPT_FRAME_I18N.get(lang, _PROMPT_FRAME_I18N["en"])
    sf = _SCENARIO_FRAME_I18N.get(lang, _SCENARIO_FRAME_I18N["en"])

    def _run(qtext, p_right):
        prefix_ids = chat_helper.build_prefix_ids(sys_prompt, device)
        user_content = frame.format(scenario=qtext)
        formatted = chat_helper.format_query_with_suffix(user_content)
        qids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([prefix_ids, qids], dim=1)
        return _b1_logit_p_spare(model, full_ids, left_id, right_id, p_right, decision_temperature)

    p1 = _run(scenario_text, pref_right)
    ll, rl = sf["left_lane"], sf["right_lane"]
    ph = "\x00S\x00"
    sw = scenario_text.replace(ll, ph).replace(rl, ll).replace(ph, rl)
    ga, gb = sf.get("group_a", "Group A"), sf.get("group_b", "Group B")
    if ga != gb:
        sw = sw.replace(ga, ph).replace(gb, ga).replace(ph, gb)
    p2 = _run(sw, not pref_right)
    return (p1 + p2) / 2.0


def run_b1_country_instruct(model, tokenizer, scenario_df, country, cfg):
    device = next(model.parameters()).device
    lang = _COUNTRY_LANG.get(country, "en")
    country_name = _COUNTRY_FULL_NAMES.get(country, country)
    sys_prompt = _B1_TEMPLATES.get(lang, _B1_TEMPLATES["en"]).format(country_name=country_name)
    chat_helper = ChatTemplateHelper(tokenizer)
    left_id = tokenizer.encode("LEFT", add_special_tokens=False)[0]
    right_id = tokenizer.encode("RIGHT", add_special_tokens=False)[0]

    rows_out = []
    for _, row in tqdm(scenario_df.iterrows(), total=len(scenario_df), desc=f"B1 [{country}]"):
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        pref_right = bool(row.get("preferred_on_right", 1))
        p_spare = _b1_two_pass(
            model=model,
            tokenizer=tokenizer,
            chat_helper=chat_helper,
            sys_prompt=sys_prompt,
            scenario_text=prompt,
            pref_right=pref_right,
            lang=lang,
            left_id=left_id,
            right_id=right_id,
            decision_temperature=cfg.decision_temperature,
            device=device,
        )
        rows_out.append(
            {
                "country": country,
                "phenomenon_category": row.get("phenomenon_category", "?"),
                "this_group_name": row.get("this_group_name", "Unknown"),
                "n_left": int(row.get("n_left", 1)),
                "n_right": int(row.get("n_right", 1)),
                "preferred_on_right": int(pref_right),
                "p_spare_preferred": float(p_spare),
            }
        )

    results_df = pd.DataFrame(rows_out)
    model_amce = compute_amce_from_preferences(results_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment = compute_alignment_metrics(model_amce, human_amce)
    out_csv = os.path.join(cfg.output_dir, f"b1_country_instruct_{country}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[B1 {country}] JSD={alignment.get('jsd', np.nan):.4f}  r={alignment.get('pearson_r', np.nan):.4f}")
    return {"country": country, "alignment": alignment, "model_amce": model_amce, "human_amce": human_amce}


def _load_model(cfg):
    from unsloth import FastLanguageModel

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
    return model, tokenizer


def main():
    cfg = SWAConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"[MODEL] Loading for B1 baseline: {cfg.model_name}")
    model, tokenizer = _load_model(cfg)

    all_results = []
    for country in cfg.target_countries:
        lang = _COUNTRY_LANG.get(country, "en")
        if cfg.use_real_data:
            base_df = load_multitp_dataset(
                cfg.multitp_data_path,
                lang=lang,
                translator=cfg.multitp_translator,
                suffix=cfg.multitp_suffix,
                n_scenarios=cfg.n_scenarios,
                seed=42,
            )
        else:
            base_df = generate_multitp_scenarios(cfg.n_scenarios, seed=42, lang=lang)

        country_df = balance_scenario_dataset(
            scenario_df=base_df,
            min_per_category=50,
            seed=42,
            lang=lang,
        )
        all_results.append(run_b1_country_instruct(model, tokenizer, country_df, country, cfg))
        torch.cuda.empty_cache()
        gc.collect()

    jsds = [r["alignment"].get("jsd", np.nan) for r in all_results]
    rs = [r["alignment"].get("pearson_r", np.nan) for r in all_results]
    print("\n" + "=" * 60)
    print(f"  BASELINE 1 — CountryInstruct | {len(all_results)} countries")
    print(f"  Mean JSD       = {np.nanmean(jsds):.4f}")
    print(f"  Mean Pearson r = {np.nanmean(rs):.4f}")
    print("=" * 60)

    pd.DataFrame([{"country": r["country"], **r["alignment"]} for r in all_results]).to_csv(
        os.path.join(cfg.output_dir, "b1_country_instruct_summary.csv"), index=False
    )
    with open(os.path.join(cfg.output_dir, "b1_country_instruct.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    print(f"[DONE] Saved B1 outputs to {cfg.output_dir}")


if __name__ == "__main__":
    main()
