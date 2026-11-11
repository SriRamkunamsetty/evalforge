"""Bootstrap confidence intervals and Bayesian uncertainty estimation for Bradley-Terry preference modeling."""
from __future__ import annotations

import math
import random
from typing import Sequence


def fit_bradley_terry(battles: Sequence[tuple[str, str]], iters: int = 150) -> dict[str, float]:
    """Fit Bradley-Terry model with Bayesian additive smoothing."""
    clean_battles = [(win, lose) for win, lose in battles if win != lose]
    if not clean_battles:
        return {}
    items = sorted({x for b in clean_battles for x in b})
    if len(items) <= 1:
        return {item: 1.0 for item in items}
        
    wins = {i: 0.5 for i in items}
    pairs: dict[frozenset[str], int] = {}
    for win, lose in clean_battles:
        wins[win] += 1
        key = frozenset((win, lose))
        pairs[key] = pairs.get(key, 0) + 1
        
    p = {i: 1.0 for i in items}
    for _ in range(iters):
        new = {}
        for i in items:
            denom = sum(
                count / (p[i] + p[next(iter(k - {i}))])
                for k, count in pairs.items()
                if i in k
            )
            new[i] = wins[i] / denom if denom > 0 else 1.0
        scale = len(items) / sum(new.values()) if sum(new.values()) > 0 else 1.0
        p = {i: v * scale for i, v in new.items()}
    return p


def elo_scale(strength: float) -> float:
    """Scale Bradley-Terry strength to standard chess Elo scale (mean ~ 1000)."""
    return 1000.0 + 400.0 * math.log10(max(strength, 1e-6))


def bradley_terry_bootstrap(
    battles: Sequence[tuple[str, str]],
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 42
) -> dict[str, dict[str, float]]:
    """Compute Bradley-Terry strengths with non-parametric bootstrap confidence intervals.
    
    Returns:
        dict[model, {
            'strength': float,
            'elo': float,
            'elo_low': float,
            'elo_high': float,
            'elo_std': float
        }]
    """
    clean_battles = [(win, lose) for win, lose in battles if win != lose]
    if not clean_battles:
        return {}
        
    rng = random.Random(seed)
    n = len(clean_battles)
    base_strengths = fit_bradley_terry(clean_battles)
    
    # Run bootstrap draws
    models = sorted(base_strengths.keys())
    elo_samples: dict[str, list[float]] = {m: [] for m in models}
    
    for _ in range(n_bootstrap):
        sample = [clean_battles[rng.randrange(n)] for _ in range(n)]
        sampled_strengths = fit_bradley_terry(sample, iters=80)
        for m in models:
            s = sampled_strengths.get(m, base_strengths[m])
            elo_samples[m].append(elo_scale(s))
            
    alpha_low = (1.0 - ci) / 2.0
    alpha_high = 1.0 - alpha_low
    
    summary: dict[str, dict[str, float]] = {}
    for m in models:
        vals = sorted(elo_samples[m])
        k = len(vals)
        mean_elo = sum(vals) / k
        variance = sum((x - mean_elo) ** 2 for x in vals) / (k - 1) if k > 1 else 0.0
        std_elo = math.sqrt(variance)
        
        low_idx = min(int(k * alpha_low), k - 1)
        high_idx = min(int(k * alpha_high), k - 1)
        
        summary[m] = {
            "strength": round(base_strengths[m], 4),
            "elo": round(elo_scale(base_strengths[m]), 1),
            "elo_low": round(vals[low_idx], 1),
            "elo_high": round(vals[high_idx], 1),
            "elo_std": round(std_elo, 2),
        }
    return summary


def compute_win_probability_matrix(strengths: dict[str, float]) -> dict[str, dict[str, float]]:
    """Compute pairwise win probability matrix P(model_i > model_j) for all models."""
    models = sorted(strengths.keys())
    matrix: dict[str, dict[str, float]] = {m: {} for m in models}
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                matrix[m1][m2] = 0.5
            else:
                p1, p2 = strengths[m1], strengths[m2]
                matrix[m1][m2] = round(p1 / (p1 + p2), 4) if (p1 + p2) > 0 else 0.5
    return matrix
