import json
import re
from collections import Counter
from typing import Optional, List, Any

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun


def extract_json(text: str) -> str:
    text = (
        re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        .replace("```", "")
        .strip()
    )

    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return text

    candidate = m.group(1).strip()

    try:
        obj = json.loads(candidate)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return candidate


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()

    if not pred_toks and not gold_toks:
        return 1.0

    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


class ChatTemplatedJsonLLM(LLM):
    pipeline: Any
    tokenizer: Any
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "hf_chat_templated_json"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        chat = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, return_full_text=False
        )

        out = self.pipeline(rendered)[0]["generated_text"]
        return extract_json(out)
