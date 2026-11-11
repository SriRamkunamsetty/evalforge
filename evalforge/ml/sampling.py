"""Active learning and uncertainty-based annotation queue prioritization."""
from __future__ import annotations

import math
from typing import Sequence, Any


def binary_entropy(p: float) -> float:
    """Compute Shannon binary entropy in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return - (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def uncertainty_sampling(
    model_a_strength: float,
    model_b_strength: float,
    existing_votes: Sequence[str] = ()
) -> float:
    """Compute an acquisition priority score for a pairwise comparison task.
    
    Higher score indicates higher uncertainty (and higher information gain if labeled).
    Tasks where models have equal strength or split votes receive highest priority.
    """
    total_strength = model_a_strength + model_b_strength
    if total_strength <= 0:
        win_prob_a = 0.5
    else:
        win_prob_a = model_a_strength / total_strength

    base_uncertainty = binary_entropy(win_prob_a)

    # Disagreement penalty/bonus: if raters have voted and are in split disagreement, prioritize adjudication
    n_votes = len(existing_votes)
    if n_votes > 0:
        a_count = existing_votes.count("a")
        b_count = existing_votes.count("b")
        
        # High contestation ratio
        contestation = (1.0 - abs(a_count - b_count) / max(n_votes, 1))
        # If task already has enough consensus votes, lower its urgency
        saturation_decay = 1.0 / (1.0 + 0.5 * n_votes)
        priority = (base_uncertainty * 0.6 + contestation * 0.4) * saturation_decay
    else:
        # Untested pair with model uncertainty
        priority = base_uncertainty

    return round(priority, 4)


def active_queue_prioritization(
    tasks: list[dict[str, Any]],
    model_strengths: dict[str, float],
    task_votes: dict[int, list[str]],
    exclude_task_ids: set[int] | None = None
) -> list[dict[str, Any]]:
    """Rank pending tasks by their active learning uncertainty score."""
    excluded = exclude_task_ids or set()
    scored_tasks = []

    for task in tasks:
        tid = task["id"]
        if tid in excluded:
            continue
            
        m_a = task.get("model_a", "")
        m_b = task.get("model_b", "")
        sa = model_strengths.get(m_a, 1.0)
        sb = model_strengths.get(m_b, 1.0)
        votes = task_votes.get(tid, [])

        score = uncertainty_sampling(sa, sb, votes)
        scored_tasks.append({
            **task,
            "uncertainty_score": score,
            "vote_count": len(votes),
        })

    # Sort descending by uncertainty score (highest information gain first)
    scored_tasks.sort(key=lambda t: t["uncertainty_score"], reverse=True)
    return scored_tasks
