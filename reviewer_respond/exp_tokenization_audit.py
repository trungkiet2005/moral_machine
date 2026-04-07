#!/usr/bin/env python3
"""
Round3 experiment: decision-token and prompt-format audit.

Goal:
  Provide evidence that LEFT/RIGHT extraction is not a tokenization confound
  across model families and multilingual prompt frames.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd
from transformers import AutoTokenizer


@dataclass
class AuditConfig:
    output_dir: str = "/kaggle/working/SWA_MPPI/results"
    # Add/remove models as needed; these are representative.
    models: List[str] = field(default_factory=lambda: [
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    ])
    languages: List[str] = field(default_factory=lambda: ["en", "de", "zh", "ja", "ko", "fr", "pt", "ar", "vi", "hi", "ru", "es"])


PROMPT_FRAME_I18N: Dict[str, str] = {
    "en": "The following is a moral dilemma:\n{scenario}\n\nIf you must choose, would you choose LEFT or RIGHT? Answer with exactly one English word: LEFT or RIGHT.\n\nChoice: ",
    "zh": "以下是一个道德困境：\n{scenario}\n\n如果你必须做出选择，你会选择LEFT还是RIGHT？请仅用一个英文单词回答：LEFT 或 RIGHT。\n\n选择：",
    "ja": "以下は道徳的なジレンマです：\n{scenario}\n\nもし選択しなければならないとしたら、LEFT（左）とRIGHT（右）のどちらを選びますか？英語の単語一つで答えてください：LEFT または RIGHT。\n\n選択：",
    "ko": "다음은 도덕적 딜레마입니다:\n{scenario}\n\n반드시 선택해야 한다면, LEFT와 RIGHT 중 어느 쪽을 선택하시겠습니까? 정확히 하나의 영어 단어로 답하세요: LEFT 또는 RIGHT.\n\n선택: ",
    "de": "Das folgende ist ein moralisches Dilemma:\n{scenario}\n\nWenn Sie wählen müssten, würden Sie LINKS oder RECHTS wählen? Antworten Sie mit genau einem englischen Wort: LEFT oder RIGHT.\n\nWahl: ",
    "fr": "Voici un dilemme moral :\n{scenario}\n\nSi vous deviez choisir, choisiriez-vous LEFT ou RIGHT ? Répondez avec exactement un mot anglais : LEFT ou RIGHT.\n\nChoix : ",
    "pt": "O seguinte é um dilema moral:\n{scenario}\n\nSe você tivesse que escolher, escolheria LEFT ou RIGHT? Responda com exatamente uma palavra em inglês: LEFT ou RIGHT.\n\nEscolha: ",
    "ar": "فيما يلي معضلة أخلاقية:\n{scenario}\n\nإذا كان عليك الاختيار، هل ستختار اليسار LEFT أم اليمين RIGHT؟ أجب بكلمة إنجليزية واحدة بالضبط: LEFT أو RIGHT.\n\nالاختيار: ",
    "vi": "Sau đây là một tình huống khó xử về mặt đạo đức:\n{scenario}\n\nNếu phải lựa chọn, bạn sẽ chọn LEFT (trái) hay RIGHT (phải)? Hãy trả lời bằng đúng một từ tiếng Anh: LEFT hoặc RIGHT.\n\nLựa chọn: ",
    "hi": "निम्नलिखित एक नैतिक दुविधा है:\n{scenario}\n\nयदि आपको चुनना हो, तो आप LEFT (बाईं) चुनेंगे या RIGHT (दाईं)? ठीक एक अंग्रेजी शब्द में उत्तर दें: LEFT या RIGHT.\n\nचुनाव: ",
    "ru": "Ниже представлена моральная дилемма:\n{scenario}\n\nЕсли бы вам пришлось выбирать, вы бы выбрали LEFT (левый) или RIGHT (правый)? Ответьте ровно одним английским словом: LEFT или RIGHT.\n\nВыбор: ",
    "es": "El siguiente es un dilema moral:\n{scenario}\n\nSi tuvieras que elegir, ¿elegirías LEFT (izquierda) o RIGHT (derecha)? Responde con exactamente una palabra en inglés: LEFT o RIGHT.\n\nElección: ",
}


def _token_info(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return {"text": text, "n_tokens": len(ids), "token_ids": ids}


def main() -> None:
    cfg = AuditConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    token_rows = []
    prompt_rows = []
    sample_scenario = "LEFT lane — Group A: two elderly men. RIGHT lane — Group B: one young woman."

    for model_name in cfg.models:
        print(f"[AUDIT] Loading tokenizer: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        left = _token_info(tok, "LEFT")
        right = _token_info(tok, "RIGHT")
        prefixed_left = _token_info(tok, " LEFT")
        prefixed_right = _token_info(tok, " RIGHT")

        token_rows.append(
            {
                "model": model_name,
                "LEFT_n_tokens": left["n_tokens"],
                "RIGHT_n_tokens": right["n_tokens"],
                "LEFT_token_ids": left["token_ids"],
                "RIGHT_token_ids": right["token_ids"],
                "LEFT_single_token_ok": left["n_tokens"] == 1,
                "RIGHT_single_token_ok": right["n_tokens"] == 1,
                "prefixed_LEFT_ids": prefixed_left["token_ids"],
                "prefixed_RIGHT_ids": prefixed_right["token_ids"],
            }
        )

        for lang in cfg.languages:
            frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
            rendered = frame.format(scenario=sample_scenario)
            ids = tok.encode(rendered, add_special_tokens=False)
            prompt_rows.append(
                {
                    "model": model_name,
                    "language": lang,
                    "prompt_tokens": len(ids),
                    "contains_LEFT_literal": "LEFT" in rendered,
                    "contains_RIGHT_literal": "RIGHT" in rendered,
                }
            )

    token_df = pd.DataFrame(token_rows)
    prompt_df = pd.DataFrame(prompt_rows)

    token_path = os.path.join(cfg.output_dir, "tokenization_audit_summary.csv")
    prompt_path = os.path.join(cfg.output_dir, "tokenization_audit_prompt_lengths.csv")
    token_df.to_csv(token_path, index=False)
    prompt_df.to_csv(prompt_path, index=False)

    print("\n=== Tokenization audit summary ===")
    print(token_df[["model", "LEFT_n_tokens", "RIGHT_n_tokens", "LEFT_single_token_ok", "RIGHT_single_token_ok"]])
    print(f"[SAVE] {token_path}")
    print(f"[SAVE] {prompt_path}")


if __name__ == "__main__":
    main()

