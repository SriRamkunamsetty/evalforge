"""EvalForge core: rater agreement, gold QA, consensus, Bradley-Terry ranking, DPO export, and ML algorithms."""
from __future__ import annotations

import hashlib
import math

# Expose advanced psychometric and ML capabilities from evalforge.ml
from .ml.agreement import gwet_ac1, krippendorff_alpha, pairwise_agreement_matrix
from .ml.bootstrap import bradley_terry_bootstrap, compute_win_probability_matrix
from .ml.judge import HeuristicJudge, JudgeComparisonResult, LLMJudge, RubricScore, calibrate_judge_against_human
from .ml.sampling import active_queue_prioritization, uncertainty_sampling
from .ml.export import DatasetFormat, export_preference_dataset, validate_preference_pair


def cohens_kappa(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def pairwise_kappas(by_rater: dict[str, dict[int, str]], min_overlap: int = 5) -> list[dict]:
    """Cohen's kappa for every rater pair with at least `min_overlap` commonly-rated tasks."""
    raters, out = sorted(by_rater), []
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            common = sorted(set(by_rater[r1]) & set(by_rater[r2]))
            if len(common) >= min_overlap:
                k = cohens_kappa([by_rater[r1][t] for t in common], [by_rater[r2][t] for t in common])
                out.append({"a": r1, "b": r2, "kappa": k, "n": len(common)})
    return out


def gold_accuracy(answers: dict, gold: dict) -> float:
    scored = [k for k in gold if k in answers]
    return sum(answers[k] == gold[k] for k in scored) / len(scored) if scored else 0.0


def flag_unreliable(accuracy_by_rater: dict[str, float], threshold: float = 0.7) -> list[str]:
    return sorted(r for r, acc in accuracy_by_rater.items() if acc < threshold)


def consensus(votes: list[str], min_votes: int = 2) -> str | None:
    """Majority of non-tie votes ('a'/'b'); None when too few votes or tied."""
    a, b = votes.count("a"), votes.count("b")
    if a + b < min_votes or a == b:
        return None
    return "a" if a > b else "b"


def swap_for(task_id: int, rater: str) -> bool:
    """Deterministic per-(task, rater) position randomisation to cancel position bias."""
    return int(hashlib.sha256(f"{task_id}:{rater}".encode()).hexdigest(), 16) % 2 == 1


def bradley_terry(battles: list[tuple[str, str]], iters: int = 200) -> dict[str, float]:
    battles = [(win, lose) for win, lose in battles if win != lose]
    items = sorted({x for b in battles for x in b})
    wins = {i: 0.5 for i in items}  # smoothing keeps strengths finite
    pairs: dict[frozenset, int] = {}
    for win, lose in battles:
        wins[win] += 1
        pairs[frozenset((win, lose))] = pairs.get(frozenset((win, lose)), 0) + 1
    p = {i: 1.0 for i in items}
    for _ in range(iters):
        new = {i: wins[i] / sum(c / (p[i] + p[next(iter(k - {i}))]) for k, c in pairs.items() if i in k)
               for i in items}
        scale = len(items) / sum(new.values())
        p = {i: v * scale for i, v in new.items()}
    return p


def elo_scale(strength: float) -> float:
    return 1000 + 400 * math.log10(strength)


def to_dpo(records: list[dict]) -> list[dict]:
    """records: {prompt, a, b, winner in {'a','b'}} -> DPO rows (ties dropped)."""
    out = []
    for r in records:
        if r.get("winner") in ("a", "b"):
            lo = "b" if r["winner"] == "a" else "a"
            out.append({"prompt": r["prompt"], "chosen": r[r["winner"]], "rejected": r[lo]})
    return out


__all__ = [
    "cohens_kappa",
    "pairwise_kappas",
    "gold_accuracy",
    "flag_unreliable",
    "consensus",
    "swap_for",
    "bradley_terry",
    "elo_scale",
    "to_dpo",
    "krippendorff_alpha",
    "gwet_ac1",
    "pairwise_agreement_matrix",
    "bradley_terry_bootstrap",
    "compute_win_probability_matrix",
    "LLMJudge",
    "HeuristicJudge",
    "RubricScore",
    "JudgeComparisonResult",
    "calibrate_judge_against_human",
    "uncertainty_sampling",
    "active_queue_prioritization",
    "export_preference_dataset",
    "DatasetFormat",
    "validate_preference_pair",
]
