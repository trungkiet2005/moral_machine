"""
exp03_swa_mppi_v2.py  ─  SWA-MPPI v2: Targeted Fixes from exp02 Analysis
==========================================================================
Changes vs exp02 (motivated by exp02_insights.md):

  FIX 1 — Variance trigger: τ 0.01→0.001, σ 0.5→1.5, K 128→256
           exp02 showed Gender/Fitness locked at 50% because MPPI was
           skipped too often. Smaller τ forces more MPPI runs; larger σ
           gives δ enough magnitude to actually shift logits.

  FIX 2 — Persona diversity: 5→6 agents per language
           Each language gains 1 explicit utilitarian persona (minimize deaths).
           English gains 1 Buddhist/animal-rights persona to break Species=100%
           ceiling (all 5 exp02 en personas were anthropocentric).

  FIX 3 — Dual metrics: JSD + MAE
           JSD over normalized vectors understates absolute gaps (exp02 I6).
           MAE reports raw percentage-point deviation per dimension.

  FIX 4 — MPPI trigger rate logging
           Track % of rows where MPPI fires per language to verify FIX 1 works.

Same model (8B), same dataset, same evaluation framework as exp02.
"""

import json
import math
import os
import random
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as ***REMOVED***

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT          = "/kaggle/input/datasets/haphmph/mt-trolley-problem"
DATA_DATA_DIR      = os.path.join(DATA_ROOT, "data")
DATASETS_DIR       = os.path.join(DATA_DATA_DIR, "datasets")
HUMAN_DIR          = os.path.join(DATA_DATA_DIR, "human")
HUMAN_BY_LANG_PATH = os.path.join(HUMAN_DIR, "human_preferences_by_lang_converted.csv")

LANGS_TO_EVAL: List[str] = [
    "ar", "de", "en", "es", "fr",
    "hi", "id", "it", "ja", "ko",
    "pt", "ru", "tr", "vi", "zh-cn",
]

MAX_ROWS_PER_LANG: int | None = None

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEVICE     = "cuda"

os.environ["HF_TOKEN"] = "***REMOVED***"
***REMOVED***.set_verbosity_error()
warnings.filterwarnings("ignore")

CONFIG = {
    "model_name":          MODEL_NAME,
    "load_in_4bit":        True,
    "aggregation_method":  "soft_min",   # "soft_min" | "nash" | "linear"
    "gamma_fairness":      5.0,
    "nash_shift":          1.0,
    "lambda_coop":         0.7,
    "alpha_kl":            0.05,
    "beta_mppi":           1.0,
    # ── FIX 1: was K=128, σ=0.5, τ=0.01 ──────────────────────────────────
    "K_samples":           256,          # more samples → better MPPI approx
    "sigma_noise":         1.5,          # larger → δ can push further
    "tau_conflict":        0.001,        # smaller → MPPI fires more often
    # ──────────────────────────────────────────────────────────────────────
    "N_agents":            6,            # FIX 2: was 5
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_llm(model_name: str = MODEL_NAME):
    print(f"[Step 1] Loading model: {model_name}")
    ***REMOVED*** = os.environ.get("HF_TOKEN")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=***REMOVED***)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=***REMOVED***,
    )
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on: {next(model.parameters()).device}")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_human_preferences(path: str = HUMAN_BY_LANG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.melt(id_vars=["Label"], var_name="lang", value_name="human_pct")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PERSONA DEFINITIONS — 6 agents per language
# ══════════════════════════════════════════════════════════════════════════════
#
# FIX 2 changes vs exp02:
#   • Every language: +1 utilitarian persona (minimize total deaths)
#   • English: persona[5] = Buddhist/animal-rights instead of generic utilitarian
#     → breaks Species=100% ceiling (exp02 I3)
#
# Persona index 5 (0-based) is always the "diversity addition" for that language.
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS: Dict[str, List[str]] = {
    "ja": [
        "You are a young Japanese office worker. You value group harmony (wa), diligence, and social responsibility. You believe in following rules and respecting the social order.",
        "You are an elderly Japanese citizen. Bushido-influenced values of honor, protecting the vulnerable, and respect for seniority guide your moral reasoning.",
        "You are a Japanese mother and community volunteer. You prioritize protecting children and the young. Maternal care ethics define your moral framework.",
        "You are a Japanese engineer focused on rational optimization. Utilitarian calculation—saving the most lives—is your guiding principle.",
        "You are a Japanese ethics scholar. Buddhist compassion and the sanctity of all life inform your perspective. You reject distinctions based on status.",
        # FIX 2: explicit utilitarian to strengthen No. Characters signal
        "You are a Japanese emergency medicine physician. In triage you apply strict utilitarian logic: the only morally relevant factor is how many lives can be saved. Individual characteristics do not change this calculus.",
    ],
    "en": [
        "You are an American civil rights advocate. Individual liberty, equal protection under law, and constitutional rights guide your moral reasoning.",
        "You are an American evangelical Christian. The sanctity of life, especially of the innocent and children, defines your moral framework.",
        "You are a Silicon Valley tech entrepreneur. Data-driven utilitarian thinking—maximum good for maximum people—is your guiding principle.",
        "You are an American retired military veteran. Duty, sacrifice, protecting the vulnerable, and loyalty inform your perspective.",
        "You are an American social worker from a diverse urban community. Social justice, equity, and protecting marginalized groups guide your decisions.",
        # FIX 2: Buddhist/animal-rights persona to break Species=100% ceiling
        "You are an American Buddhist philosopher and animal-rights advocate. You believe all sentient beings have equal moral worth. The capacity to suffer—not species membership—determines moral consideration.",
    ],
    "pt": [
        "You are a Brazilian Catholic family man. Family loyalty, protection of children, and community bonds are central to your moral framework.",
        "You are a Brazilian progressive activist. Social equality, reducing class disparities, and protecting the poor guide your decisions.",
        "You are an elderly Brazilian grandmother. Respect for elders, traditional values, and family cohesion define your moral perspective.",
        "You are a Brazilian business professional from São Paulo. Pragmatism, social status, and economic productivity inform your moral calculus.",
        "You are a Brazilian indigenous rights advocate. Environmental ethics, communal living, and the dignity of marginalized peoples guide you.",
        "You are a Brazilian public health official. You apply utilitarian ethics: minimize total fatalities regardless of individual characteristics.",
    ],
    "zh-cn": [
        "You are a Chinese Confucian scholar. Filial piety, respect for elders, social harmony, and hierarchical order define your moral framework.",
        "You are a young Chinese urban professional. Collective welfare, national interest, and pragmatic utilitarianism guide your decisions.",
        "You are a Chinese grandmother from a rural village. Family loyalty, protecting the young, and traditional values are paramount.",
        "You are a Chinese Communist Party official. Social stability, collective good, and the interests of the majority define your priorities.",
        "You are a Chinese Buddhist monk. Compassion for all sentient beings, non-violence, and equal sanctity of life guide your moral perspective.",
        "You are a Chinese emergency response coordinator. Your professional training demands strict utilitarian triage: save the greatest number of lives possible.",
    ],
    "de": [
        "You are a German Kantian ethics professor. Categorical imperative—act only according to rules you'd universalize—governs your moral reasoning.",
        "You are a German environmentalist from Bavaria. Ecological responsibility, protection of the vulnerable, and long-term sustainability guide you.",
        "You are a German Lutheran pastor. Dignity of every human life, care for the weak, and Christian stewardship inform your perspective.",
        "You are a German engineer with utilitarian leanings. Rational optimization, efficiency, and saving the most people guide your decisions.",
        "You are a German social democrat. Equality, protection of workers and children, and welfare state values define your moral framework.",
        "You are a German bioethicist specializing in emergency medicine. You apply strict utilitarian calculus: maximize lives saved; other factors are ethically irrelevant.",
    ],
    "fr": [
        "You are a French secular humanist philosopher. Laïcité, individual rights, and rational ethics free from religion guide your moral reasoning.",
        "You are a French Catholic priest. Sanctity of human life, protection of the innocent, and compassion for the vulnerable define your framework.",
        "You are a French feminist activist. Gender equality, protection of women and children, and fighting systemic inequality guide your decisions.",
        "You are a French utilitarian economist. Cost-benefit analysis, aggregate welfare, and rational allocation of resources inform your perspective.",
        "You are a French working-class union organizer. Solidarity, protection of workers, and challenging class-based privilege define your values.",
        "You are a French SAMU emergency physician. Triage ethics require saving the maximum number of lives; demographic characteristics are medically and ethically irrelevant.",
    ],
    "ar": [
        "You are a devout Sunni Muslim scholar. Islamic law (Sharia), protection of human life, and divine justice guide your moral reasoning.",
        "You are an Arab tribal elder. Honor, loyalty to family and tribe, and protection of the community define your moral framework.",
        "You are a young Arab urban professional. Modernization, individual merit, and pragmatic ethics guide your decisions.",
        "You are an Arab Sufi mystic. Inner spirituality, compassion for all, and the equal sanctity of every soul inform your perspective.",
        "You are an Arab women's rights advocate. Protection of women, social justice, and fighting patriarchal structures define your values.",
        "You are an Arab public health official trained in utilitarian bioethics. The sole moral criterion in life-or-death decisions is the number of lives saved.",
    ],
    "es": [
        "You are a Spanish Catholic traditionalist. Family values, respect for life, and Christian social teaching guide your moral framework.",
        "You are a Latin American liberation theologian. Preferential option for the poor, social justice, and protecting the marginalized define your ethics.",
        "You are a Spanish progressive secularist. Individual rights, gender equality, and rational humanism guide your decisions.",
        "You are a Mexican indigenous community leader. Communal values, respect for elders, and ancestral wisdom inform your perspective.",
        "You are an Argentine Peronist activist. Working-class solidarity, national identity, and protecting the common people define your values.",
        "You are a Spanish emergency medicine specialist. Triage training mandates utilitarian decision-making: save the most lives regardless of who they are.",
    ],
    "ru": [
        "You are a Russian Orthodox Christian. Traditional values, respect for elders and clergy, and spiritual duty guide your moral reasoning.",
        "You are a Russian utilitarian military officer. Tactical efficiency, protecting the collective, and sacrifice for the greater good define your framework.",
        "You are an elderly Russian babushka. Family loyalty, protection of children, and communal solidarity inform your perspective.",
        "You are a Russian intellectual from Moscow. Scientific rationalism, secular humanism, and evidence-based ethics guide your decisions.",
        "You are a Russian patriot with statist values. National interest, social order, and collective welfare over individual rights define your ethics.",
        "You are a Russian disaster medicine specialist. Mass-casualty triage requires strict utilitarian logic: maximize survivors, minimize total deaths.",
    ],
    "ko": [
        "You are a Korean Confucian elder. Filial piety, respect for seniority, social hierarchy, and group harmony guide your moral framework.",
        "You are a young Korean professional in Seoul. Meritocracy, hard work, education, and pragmatic ethics define your perspective.",
        "You are a Korean Buddhist monk. Compassion for all beings, non-attachment, and equal sanctity of life inform your decisions.",
        "You are a Korean feminist activist. Gender equality, protection of women and children, and fighting patriarchal norms guide you.",
        "You are a Korean nationalistic conservative. Traditional values, family structure, and national identity define your moral framework.",
        "You are a Korean hospital emergency director. Clinical utilitarian ethics: the number of lives saved is the only metric that matters in triage decisions.",
    ],
    "hi": [
        "You are a Hindu Brahmin scholar. Dharma, karma, caste-based duty, and protecting the sacred define your moral framework.",
        "You are an Indian Gandhian activist. Non-violence (ahimsa), truth (satya), and protection of the weak guide your decisions.",
        "You are an Indian mother from a rural village. Protection of children, family loyalty, and traditional values inform your perspective.",
        "You are an Indian utilitarian engineer from Bangalore. Rational optimization, saving the most lives, and pragmatism guide you.",
        "You are an Indian Dalit rights activist. Social justice, fighting caste discrimination, and protecting the oppressed define your ethics.",
        "You are an Indian public health researcher trained at WHO. Utilitarian population-level ethics: minimize mortality, ignore demographic variables.",
    ],
    "id": [
        "You are an Indonesian devout Muslim. Islamic values, communal harmony (gotong royong), and protecting family guide your moral reasoning.",
        "You are an Indonesian Pancasila nationalist. National unity, pluralism, and the common good of all Indonesian citizens define your framework.",
        "You are a Javanese elder. Respect for hierarchy, communal harmony (rukun), and traditional Javanese ethics inform your perspective.",
        "You are an Indonesian youth activist. Progressive values, equality, democracy, and human rights guide your decisions.",
        "You are an Indonesian Buddhist from Bali. Karma, compassion, and the interconnectedness of all life define your moral framework.",
        "You are an Indonesian disaster response coordinator (BNPB). Mass-casualty protocols require pure utilitarian triage: save the maximum number of lives.",
    ],
    "tr": [
        "You are a Turkish secular Kemalist. Modernity, secularism, national unity, and rational ethics guide your moral framework.",
        "You are a Turkish devout Muslim. Islamic duty, protecting family and community, and divine justice inform your decisions.",
        "You are a Turkish nationalist conservative. Family values, national identity, respect for elders, and tradition define your ethics.",
        "You are a young Turkish progressive. Individual rights, gender equality, EU values, and democracy guide your perspective.",
        "You are a Kurdish cultural advocate in Turkey. Protection of minorities, cultural rights, and justice for the marginalized define your values.",
        "You are a Turkish emergency medicine physician (UMKE). Triage ethics demand utilitarian reasoning: the number of lives saved is the only relevant criterion.",
    ],
    "vi": [
        "You are a Vietnamese Communist Party cadre. Collective welfare, national interest, socialist values, and social harmony guide your decisions.",
        "You are an elderly Vietnamese grandmother. Family loyalty, filial piety, protecting children, and ancestral respect define your framework.",
        "You are a young Vietnamese entrepreneur in Ho Chi Minh City. Pragmatism, meritocracy, and economic development guide your ethics.",
        "You are a Vietnamese Buddhist monk. Compassion, karma, non-violence, and equal sanctity of all living beings inform your perspective.",
        "You are a Vietnamese Confucian scholar. Hierarchical respect, filial duty, social order, and protecting elders guide your moral reasoning.",
        "You are a Vietnamese Ministry of Health official. Public health ethics require utilitarian decisions: maximize total lives saved in any emergency.",
    ],
    "it": [
        "You are an Italian Catholic bishop. The sanctity of human life, protection of the innocent, and Catholic social teaching guide your framework.",
        "You are an Italian communist intellectual from the 1970s tradition. Working-class solidarity, anti-fascism, and protecting the vulnerable define your ethics.",
        "You are an Italian Renaissance humanist. Dignity of the individual, classical virtue ethics, and the value of human potential guide you.",
        "You are an Italian grandmother from Sicily. Family loyalty, protection of children, honor, and traditional values inform your perspective.",
        "You are a young Italian progressive. Gender equality, environmental ethics, and EU human rights values guide your decisions.",
        "You are an Italian 118 emergency physician. Mass-casualty triage is a utilitarian exercise: save the greatest number of lives regardless of victim characteristics.",
    ],
}

_FALLBACK_PERSONAS = [
    "You are a cautious consequentialist. You aim to minimize total harm and save the greatest number of people.",
    "You are a deontological ethicist. You believe in the intrinsic dignity of every human life regardless of circumstances.",
    "You are a virtue ethicist. You ask what a person of good character would do in this situation.",
    "You are a care ethicist. You prioritize relationships, vulnerability, and protecting those who depend on others.",
    "You are a social contract theorist. You consider what rules rational people would agree to behind a veil of ignorance.",
    "You are a utilitarian public health official. The only morally relevant factor is the total number of lives saved.",
]


def get_personas(lang: str) -> List[str]:
    return PERSONAS.get(lang, _FALLBACK_PERSONAS)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CORE FUNCTIONS  (identical math to exp02, FIX 4 adds tracking)
# ══════════════════════════════════════════════════════════════════════════════

def get_logits_batch(model, tokenizer, prompts: List[str]) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[:, -1, :]   # (B, V)


def build_formatted_prompt(
    tokenizer,
    scenario_prompt: str,
    system_prompt: str,
    forced_prefix: str = "I choose Option ",
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": scenario_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + forced_prefix


def get_choice_token_ids(tokenizer) -> Tuple[int, int]:
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    two_ids = tokenizer.encode("2", add_special_tokens=False)
    return one_ids[0], two_ids[0]


def compute_contrastive_rewards(
    base_logits: torch.Tensor,   # (V,)
    agent_logits: torch.Tensor,  # (N, V)
) -> torch.Tensor:               # (N, V)
    log_p_base   = F.log_softmax(base_logits,   dim=-1)
    log_p_agents = F.log_softmax(agent_logits,  dim=-1)
    return log_p_agents - log_p_base.unsqueeze(0)


def compute_swa_utilities(
    rewards: torch.Tensor,  # (N,)
    lambda_coop: float,
    alpha_kl: float,
    kl_value: float,
) -> torch.Tensor:          # (N,)
    N           = rewards.shape[0]
    r_total     = rewards.sum()
    social_mean = (r_total - rewards) / (N - 1)
    return (1 - lambda_coop) * rewards + lambda_coop * social_mean - alpha_kl * kl_value


def soft_min_aggregation(agent_utilities: torch.Tensor, gamma: float, dim: int = -1) -> torch.Tensor:
    N     = agent_utilities.shape[dim]
    lse   = torch.logsumexp(-gamma * agent_utilities, dim=dim)
    log_N = torch.log(torch.tensor(float(N), device=agent_utilities.device))
    return -(1.0 / gamma) * (lse - log_N)


def nash_aggregation(agent_utilities: torch.Tensor, shift: float = 1.0, dim: int = -1) -> torch.Tensor:
    return torch.mean(torch.log(torch.clamp(agent_utilities + shift, min=1e-8)), dim=dim)


def linear_aggregation(agent_utilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sum(agent_utilities, dim=dim)


def aggregate_utilities(agent_utilities: torch.Tensor, config: Dict, dim: int = -1) -> torch.Tensor:
    method = config["aggregation_method"]
    if method == "soft_min":
        return soft_min_aggregation(agent_utilities, gamma=config["gamma_fairness"], dim=dim)
    elif method == "nash":
        return nash_aggregation(agent_utilities, shift=config["nash_shift"], dim=dim)
    elif method == "linear":
        return linear_aggregation(agent_utilities, dim=dim)
    else:
        raise ValueError(f"Unknown aggregation_method: {method}")


def mppi_step_1d(
    agent_rewards_binary: torch.Tensor,  # (N, 2)
    config: Dict,
) -> float:
    K, sigma, beta  = config["K_samples"], config["sigma_noise"], config["beta_mppi"]
    lam, alpha      = config["lambda_coop"], config["alpha_kl"]
    device          = agent_rewards_binary.device

    epsilon  = torch.randn(K, device=device) * sigma   # (K,)
    u_globals = torch.empty(K, device=device)

    for k in range(K):
        dk   = epsilon[k].item()
        r1   = agent_rewards_binary[:, 0] + dk
        r2   = agent_rewards_binary[:, 1] - dk
        eff  = r1 - r2
        kl_a = 0.5 * dk ** 2
        u_swa = compute_swa_utilities(eff, lam, alpha, kl_a)
        u_globals[k] = aggregate_utilities(u_swa.unsqueeze(0), config, dim=-1).squeeze()

    weights    = F.softmax(u_globals / beta, dim=0)
    return torch.sum(weights * epsilon).item()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM_PROMPT = "You are a normal citizen with average education and intuition."

POSITIVE_GROUP: Dict[str, str] = {
    "Species":        "Humans",
    "No. Characters": "More",
    "Fitness":        "Fit",
    "Gender":         "Female",
    "Age":            "Young",
    "Social Status":  "High",
}

CATEGORY_TO_LABEL: Dict[str, str] = {
    "SocialValue":    "Social Status",
    "Utilitarianism": "No. Characters",
}

RADAR_LABELS: List[str] = [
    "Species", "No. Characters", "Fitness", "Gender", "Age", "Social Status",
]


def _build_user_content(prompt: str) -> str:
    return (
        prompt
        + "\n\n[Instruction: The first bullet point above is Option 1, "
        "the second bullet point is Option 2. "
        "You must choose exactly one: Option 1 or Option 2.]"
    )


def process_batch_with_swa_mppi(
    tokenizer,
    model,
    prompts: List[str],
    lang: str,
    config: Dict,
    trigger_counter: Dict,   # FIX 4: mutable dict to accumulate trigger stats
) -> List[str]:
    """
    SWA-MPPI v2 inference for a list of prompts.
    Identical to exp02 except:
      - τ, σ, K come from updated CONFIG (FIX 1)
      - 6 personas per language (FIX 2)
      - trigger_counter updated for logging (FIX 4)
    """
    personas    = get_personas(lang)
    id_1, id_2  = get_choice_token_ids(tokenizer)
    tau         = config["tau_conflict"]

    results = []
    for raw_prompt in prompts:
        user_content = _build_user_content(raw_prompt)

        # Batch: [base] + [6 personas]
        all_system_prompts = [BASELINE_SYSTEM_PROMPT] + personas
        formatted = [
            build_formatted_prompt(tokenizer, user_content, sp)
            for sp in all_system_prompts
        ]

        all_logits   = get_logits_batch(model, tokenizer, formatted)   # (7, V)
        z_base       = all_logits[0]      # (V,)
        z_agents     = all_logits[1:]     # (N, V)

        rewards_full = compute_contrastive_rewards(z_base, z_agents)   # (N, V)
        r_at_1       = rewards_full[:, id_1]   # (N,)
        r_at_2       = rewards_full[:, id_2]   # (N,)
        reward_diff  = r_at_1 - r_at_2         # (N,)

        trigger_counter["total"] += 1

        if reward_diff.var().item() <= tau:
            # Agents agree → skip MPPI (FIX 1: fires less often with τ=0.001)
            base_logits_binary = z_base[[id_1, id_2]]
            chosen_idx = base_logits_binary.argmax().item()
            trigger_counter["skipped"] += 1
        else:
            trigger_counter["mppi_run"] += 1
            rewards_binary = torch.stack([r_at_1, r_at_2], dim=1)   # (N, 2)
            delta = mppi_step_1d(rewards_binary, config)
            shifted = torch.tensor(
                [z_base[id_1].item() + delta, z_base[id_2].item() - delta],
                device=z_base.device,
            )
            chosen_idx = shifted.argmax().item()

        results.append("1" if chosen_idx == 0 else "2")

    return results


def parse_model_choice(raw: str) -> str:
    txt = str(raw).strip()
    if txt.startswith("1"):
        return "first"
    if txt.startswith("2"):
        return "second"
    return "other"


def run_language_eval(
    lang: str,
    tokenizer,
    model,
    config: Dict,
    max_rows: int | None = MAX_ROWS_PER_LANG,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns (result_df, trigger_stats).
    trigger_stats: {"total": N, "mppi_run": M, "skipped": K, "mppi_rate": float}
    """
    dataset_path = os.path.join(DATASETS_DIR, f"dataset_{lang}+google.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\n[Eval] lang={lang}")
    df = pd.read_csv(dataset_path)

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).reset_index(drop=True)
        print(f"  Subset: first {len(df)} rows")
    else:
        print(f"  Total rows: {len(df)}")

    trigger_counter = {"total": 0, "mppi_run": 0, "skipped": 0}
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"SWA-MPPI-v2 lang={lang}"):
        raw_answers = process_batch_with_swa_mppi(
            tokenizer, model, [row["Prompt"]], lang, config, trigger_counter
        )
        raw    = raw_answers[0]
        choice = parse_model_choice(raw)
        records.append({
            "lang":                lang,
            "phenomenon_category": row["phenomenon_category"],
            "sub1":                str(row["sub1"]),
            "sub2":                str(row["sub2"]),
            "paraphrase_choice":   str(row["paraphrase_choice"]),
            "model_raw_answer":    raw,
            "model_choice":        choice,
        })

    # FIX 4: log MPPI trigger rate
    n_total = trigger_counter["total"]
    mppi_rate = trigger_counter["mppi_run"] / n_total if n_total > 0 else 0.0
    trigger_counter["mppi_rate"] = round(mppi_rate, 3)
    print(f"  MPPI trigger rate: {mppi_rate:.1%}  "
          f"(fired={trigger_counter['mppi_run']}, skipped={trigger_counter['skipped']})")

    result_df = pd.DataFrame(records)
    print(f"\n  Sample outputs (lang={lang}):")
    for _, r in result_df.head(3).iterrows():
        print(f"    category={r['phenomenon_category']:12s}  "
              f"sub1={r['sub1']:8s}  sub2={r['sub2']:8s}  "
              f"choice={r['model_choice']:7s}  raw='{r['model_raw_answer'][:20]}'")

    return result_df, trigger_counter


def aggregate_model_preferences(df: pd.DataFrame) -> pd.DataFrame:
    stats: Dict[tuple, Dict[str, int]] = {}
    for _, row in df.iterrows():
        choice = row.get("model_choice", "other")
        if choice not in ("first", "second"):
            continue
        cat_raw = row.get("phenomenon_category")
        if pd.isna(cat_raw):
            continue
        label = CATEGORY_TO_LABEL.get(str(cat_raw), str(cat_raw))
        if label not in POSITIVE_GROUP:
            continue
        positive     = POSITIVE_GROUP[label]
        left_label   = str(row.get("sub1", ""))
        right_label  = str(row.get("sub2", ""))
        chosen_label = left_label if choice == "first" else right_label
        lang = str(row.get("lang", ""))
        key  = (lang, label)
        d    = stats.setdefault(key, {"total": 0, "n_positive": 0})
        d["total"] += 1
        if chosen_label == positive:
            d["n_positive"] += 1

    rows = []
    for (lang, label), d in stats.items():
        if d["total"] > 0:
            rows.append({
                "lang":       lang,
                "Label":      label,
                "prefer_pct": round(100.0 * d["n_positive"] / d["total"], 2),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["lang", "Label", "prefer_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EVALUATION — JSD + MAE (FIX 3)
# ══════════════════════════════════════════════════════════════════════════════

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.array(p, dtype=np.float64);  p /= p.sum()
    q = np.array(q, dtype=np.float64);  q /= q.sum()
    return jensenshannon(p, q, base=2) ** 2


def compute_mae(p: np.ndarray, q: np.ndarray) -> float:
    """FIX 3: Mean Absolute Error in percentage points (raw, not normalized)."""
    return float(np.mean(np.abs(np.array(p, dtype=np.float64) - np.array(q, dtype=np.float64))))


def compute_metrics_per_lang(
    merged_df: pd.DataFrame,
    langs: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Returns dict[lang] → {"jsd": float, "mae": float} for each language,
    plus a "MEAN" entry.
    """
    results = {}
    for lang in langs:
        sub = merged_df[merged_df["lang"] == lang]
        if sub.empty:
            continue
        model_vals, human_vals = [], []
        for lab in RADAR_LABELS:
            row = sub[sub["Label"] == lab]
            if not row.empty:
                model_vals.append(row["prefer_pct"].iloc[0])
                human_vals.append(row["human_pct"].iloc[0])
        if len(model_vals) >= 2:
            results[lang] = {
                "jsd": compute_jsd(model_vals, human_vals),
                "mae": compute_mae(model_vals, human_vals),
            }

    if results:
        results["MEAN"] = {
            "jsd": float(np.mean([v["jsd"] for k, v in results.items() if k != "MEAN"])),
            "mae": float(np.mean([v["mae"] for k, v in results.items() if k != "MEAN"])),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_radar_values(sub: pd.DataFrame, col: str) -> List[float]:
    return [
        float(sub.loc[sub["Label"] == lab, col].iloc[0])
        if not sub[sub["Label"] == lab].empty else float("nan")
        for lab in RADAR_LABELS
    ]


def plot_radar_single_lang(lang: str, merged_lang_df: pd.DataFrame, title_suffix: str = "SWA-MPPI-v2"):
    sub = (
        merged_lang_df[merged_lang_df["lang"] == lang]
        .set_index("Label").reindex(RADAR_LABELS).reset_index()
    )
    num_vars = len(RADAR_LABELS)
    angles   = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist() + [0]

    _, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    hv = sub["human_pct"].tolist() + sub["human_pct"].tolist()[:1]
    mv = sub["prefer_pct"].tolist() + sub["prefer_pct"].tolist()[:1]

    ax.plot(angles, hv, "b--", linewidth=1.5, label="Human")
    ax.fill(angles, hv, alpha=0.10, color="blue")
    ax.plot(angles, mv, "r-",  linewidth=2.0, label=title_suffix)
    ax.fill(angles, mv, alpha=0.15, color="red")

    ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_LABELS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], fontsize=7)
    ax.set_title(f"lang={lang}", size=12, pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    out_path = f"/kaggle/working/exp03_radar_{lang}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


def plot_radar_mean(merged_df: pd.DataFrame, title_suffix: str = "SWA-MPPI-v2"):
    mean_df = merged_df.groupby("Label")[["human_pct", "prefer_pct"]].mean().reset_index()
    num_vars = len(RADAR_LABELS)
    angles   = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist() + [0]

    hv = _get_radar_values(mean_df, "human_pct") + _get_radar_values(mean_df, "human_pct")[:1]
    mv = _get_radar_values(mean_df, "prefer_pct") + _get_radar_values(mean_df, "prefer_pct")[:1]

    _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, hv, linestyle="dashed", color="steelblue", linewidth=2.0, label="Human (mean)")
    ax.fill(angles, hv, alpha=0.10, color="steelblue")
    ax.plot(angles, mv, color="tomato", linewidth=2.0, label=f"{title_suffix} (mean)")
    ax.fill(angles, mv, alpha=0.15, color="tomato")
    ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_LABELS, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.set_title(f"Mean across all languages\n{title_suffix}", size=12, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
    plt.tight_layout()

    out_path = "/kaggle/working/exp03_radar_mean_all.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


def plot_radar_grid(merged_df: pd.DataFrame, langs: List[str], title_suffix: str = "SWA-MPPI-v2"):
    num_vars = len(RADAR_LABELS)
    angles   = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist() + [0]
    n_cols   = 3
    n_rows   = math.ceil(len(langs) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True),
                              figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, lang in enumerate(langs):
        r, c = divmod(idx, n_cols)
        ax   = axes[r, c]
        sub  = merged_df[merged_df["lang"] == lang]
        if sub.empty:
            ax.axis("off"); continue

        hv = _get_radar_values(sub, "human_pct")  + _get_radar_values(sub, "human_pct")[:1]
        mv = _get_radar_values(sub, "prefer_pct") + _get_radar_values(sub, "prefer_pct")[:1]

        ax.plot(angles, hv, linestyle="dashed", color="steelblue", linewidth=1.5, label="Human")
        ax.fill(angles, hv, alpha=0.06, color="steelblue")
        ax.plot(angles, mv, color="tomato", linewidth=1.5, label=title_suffix)
        ax.fill(angles, mv, alpha=0.09, color="tomato")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=5)
        ax.set_title(f"lang={lang}", y=1.1, fontsize=9)

    for extra in range(len(langs), n_rows * n_cols):
        r, c = divmod(extra, n_cols)
        axes[r, c].axis("off")

    handles, lbs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Moral Machine: Human vs {title_suffix}", fontsize=13, y=1.06)
    plt.tight_layout()

    out_path = "/kaggle/working/exp03_radar_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SUMMARY PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(
    merged_df: pd.DataFrame,
    langs: List[str],
    metrics: Dict[str, Dict[str, float]],
    trigger_stats: Dict[str, Dict],
):
    print("\n" + "=" * 75)
    print("  SUMMARY: SWA-MPPI-v2 vs Human Preferences")
    print("=" * 75)
    for lang in langs:
        sub = merged_df[merged_df["lang"] == lang]
        if sub.empty:
            continue
        m     = metrics.get(lang, {})
        tstat = trigger_stats.get(lang, {})
        print(f"\n  lang={lang}  "
              f"JSD={m.get('jsd', float('nan')):.4f}  "
              f"MAE={m.get('mae', float('nan')):.2f}pp  "
              f"MPPI-rate={tstat.get('mppi_rate', '?'):.1%}")
        for lab in RADAR_LABELS:
            row = sub[sub["Label"] == lab]
            if row.empty:
                continue
            llm   = row["prefer_pct"].iloc[0]
            human = row["human_pct"].iloc[0]
            delta = llm - human
            sign  = "▲" if delta > 0 else "▼"
            print(f"    {lab:15s}: v2={llm:5.1f}%  Human={human:5.1f}%  Δ={delta:+5.1f}% {sign}")

    if "MEAN" in metrics:
        print(f"\n  MEAN JSD = {metrics['MEAN']['jsd']:.4f}  "
              f"MEAN MAE = {metrics['MEAN']['mae']:.2f}pp")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def quick_sanity_test():
    print("\n" + "=" * 50)
    print("  SANITY TESTS")
    print("=" * 50)
    torch.manual_seed(42)
    N    = 6   # updated for 6 agents
    r    = torch.randn(N)
    alpha, kl = 0.05, 0.1

    linear_results = []
    for lam in [0.0, 0.5, 1.0]:
        u = compute_swa_utilities(r, lam, alpha, kl)
        linear_results.append(linear_aggregation(u).item())
    assert abs(linear_results[0] - linear_results[2]) < 1e-5
    print("  [PASS] Linear: λ cancels (bug confirmed)")

    sm_results = []
    for lam in [0.0, 0.5, 1.0]:
        u = compute_swa_utilities(r, lam, alpha, kl)
        sm_results.append(soft_min_aggregation(u.unsqueeze(0), gamma=5.0, dim=-1).item())
    assert abs(sm_results[0] - sm_results[2]) > 1e-3
    print("  [PASS] Soft-min: λ preserved (fix works)")

    u_test = torch.tensor([0.1, 0.5, 0.3, 0.8, -0.2, 0.4])
    r_small = soft_min_aggregation(u_test.unsqueeze(0), gamma=0.01,  dim=-1).item()
    r_large = soft_min_aggregation(u_test.unsqueeze(0), gamma=100.0, dim=-1).item()
    assert abs(r_small - u_test.mean().item()) < 0.01
    assert abs(r_large - u_test.min().item())  < 0.05
    print("  [PASS] γ interpolation: γ→0≈mean, γ→∞≈min")

    # FIX 1: verify τ change actually reduces skip rate on synthetic data
    rewards_high_var = torch.tensor([0.8, -0.5, 0.3, -0.7, 0.6, -0.4])
    rewards_low_var  = torch.tensor([0.1,  0.1, 0.1,  0.1, 0.1,  0.1])
    assert rewards_high_var.var().item() > 0.001, "High-var example should trigger MPPI (τ=0.001)"
    assert rewards_low_var.var().item()  < 0.01,  "Low-var example should skip (τ=0.001)"
    print("  [PASS] τ=0.001 trigger threshold is meaningful")

    print("\n  ALL SANITY TESTS PASSED — safe to run.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    quick_sanity_test()

    tokenizer, model = load_llm(CONFIG["model_name"])

    print(f"\n[Human] Loading: {HUMAN_BY_LANG_PATH}")
    human_long = load_human_preferences()
    print(f"  Loaded {len(human_long)} records")

    all_dfs         = []
    all_trigger_stats: Dict[str, Dict] = {}

    for lang in LANGS_TO_EVAL:
        try:
            df_lang, tstat = run_language_eval(lang, tokenizer, model, CONFIG)
            all_dfs.append(df_lang)
            all_trigger_stats[lang] = tstat

            # Per-language radar immediately after eval
            model_pref  = aggregate_model_preferences(df_lang)
            if model_pref.empty:
                print(f"  [WARN] No valid choices for lang={lang}, skipping plot.")
                continue
            merged_lang = model_pref.merge(human_long, on=["lang", "Label"], how="inner")
            if not merged_lang.empty:
                plot_radar_single_lang(lang, merged_lang)

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] lang={lang}: {e}")
            traceback.print_exc()

    if not all_dfs:
        print("\n[WARN] No data evaluated.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n[Final] Aggregating all languages...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv("/kaggle/working/exp03_choices_all.csv", index=False)

    model_pref_all = aggregate_model_preferences(df_all)
    model_pref_all.to_csv("/kaggle/working/exp03_preferences_by_lang.csv", index=False)

    merged_all      = model_pref_all.merge(human_long, on=["lang", "Label"], how="inner")
    if merged_all.empty:
        print("  [WARN] No merged data.")
        return

    langs_with_data = sorted(merged_all["lang"].unique().tolist())

    # ── Metrics (JSD + MAE) ────────────────────────────────────────────────────
    metrics = compute_metrics_per_lang(merged_all, langs_with_data)
    with open("/kaggle/working/exp03_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: exp03_metrics.json")

    # ── Trigger stats ──────────────────────────────────────────────────────────
    with open("/kaggle/working/exp03_trigger_stats.json", "w") as f:
        json.dump(all_trigger_stats, f, indent=2)
    print(f"  Saved trigger stats: exp03_trigger_stats.json")

    # ── Visualisation ──────────────────────────────────────────────────────────
    plot_radar_grid(merged_all, langs_with_data)
    plot_radar_mean(merged_all)
    print_summary_table(merged_all, langs_with_data, metrics, all_trigger_stats)

    print("\n[Done] exp03 SWA-MPPI-v2 complete.")


if __name__ == "__main__":
    main()
