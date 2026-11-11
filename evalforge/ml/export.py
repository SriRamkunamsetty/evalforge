"""Production-grade RLHF/DPO/KTO dataset export engine with quality gates."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Sequence


class DatasetFormat(str, Enum):
    DPO = "dpo"
    KTO = "kto"
    CHATML = "chatml"
    HUGGINGFACE = "hf"


def validate_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    min_char_len: int = 5,
    max_similarity: float = 0.98
) -> tuple[bool, str]:
    """Quality gate for preference pairs before inclusion in alignment training datasets."""
    p_strip, c_strip, r_strip = prompt.strip(), chosen.strip(), rejected.strip()
    if len(p_strip) < min_char_len:
        return False, "Prompt is too short"
    if len(c_strip) < min_char_len or len(r_strip) < min_char_len:
        return False, "Completion is too short"
    if c_strip == r_strip:
        return False, "Chosen and rejected responses are identical"
        
    # Lexical overlap sanity check
    words_c = set(c_strip.lower().split())
    words_r = set(r_strip.lower().split())
    if words_c and words_r:
        jaccard = len(words_c & words_r) / len(words_c | words_r)
        if jaccard > max_similarity and len(c_strip) == len(r_strip):
            return False, "Trivial lexical difference between completions"

    return True, "Valid"


def export_preference_dataset(
    records: Sequence[dict[str, Any]],
    output_format: DatasetFormat = DatasetFormat.DPO,
    filter_invalid: bool = True
) -> list[dict[str, Any]]:
    """Transform annotated battles into standardized alignment dataset records.
    
    Expected record dict structure:
    {
        'prompt': str,
        'model_a': str,
        'response_a': str,
        'model_b': str,
        'response_b': str,
        'winner': 'a' | 'b',
        'issues': list[str], optional
        'metadata': dict, optional
    }
    """
    output = []

    for r in records:
        winner = r.get("winner")
        if winner not in ("a", "b"):
            continue  # Exclude ties or unresolved battles

        prompt = r["prompt"]
        chosen = r["response_a"] if winner == "a" else r["response_b"]
        rejected = r["response_b"] if winner == "a" else r["response_a"]
        chosen_model = r["model_a"] if winner == "a" else r["model_b"]
        rejected_model = r["model_b"] if winner == "a" else r["model_a"]

        if filter_invalid:
            is_valid, _ = validate_preference_pair(prompt, chosen, rejected)
            if not is_valid:
                continue

        if output_format == DatasetFormat.DPO:
            output.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
            })
        elif output_format == DatasetFormat.KTO:
            # KTO decomposes pairwise preferences into binary desirability examples
            output.append({
                "prompt": prompt,
                "completion": chosen,
                "label": True,
                "model": chosen_model
            })
            output.append({
                "prompt": prompt,
                "completion": rejected,
                "label": False,
                "model": rejected_model
            })
        elif output_format == DatasetFormat.CHATML:
            output.append({
                "messages_chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen}
                ],
                "messages_rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected}
                ],
                "metadata": {
                    "chosen_model": chosen_model,
                    "rejected_model": rejected_model
                }
            })
        elif output_format == DatasetFormat.HUGGINGFACE:
            output.append({
                "instruction": prompt,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
                "issues_tagged": r.get("issues", []),
                "task_id": r.get("task_id"),
            })

    return output


def to_jsonl_stream(dataset: Sequence[dict[str, Any]]) -> str:
    """Serialize dataset records to standard newline-delimited JSON (JSONL)."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset)
