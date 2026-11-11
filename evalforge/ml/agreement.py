"""Advanced inter-annotator agreement metrics: Cohen's Kappa, Gwet's AC1, and Krippendorff's Alpha."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence, Any, Literal


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Compute Cohen's Kappa between two raters over identical units.
    
    Raises ValueError if length mismatch or empty sequences.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    categories = sorted(set(a) | set(b))
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in categories)
    if math.isclose(pe, 1.0, abs_tol=1e-9):
        return 1.0
    return (po - pe) / (1.0 - pe)


def gwet_ac1(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Compute Gwet's AC1 agreement statistic.
    
    Unlike Cohen's Kappa, Gwet's AC1 is resistant to the 'high prevalence' paradox,
    where high observed agreement yields paradoxically low or negative kappa values.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    categories = list(set(a) | set(b))
    q = len(categories)
    if q <= 1:
        return 1.0

    # Probability of chance agreement under Gwet's model
    # pi_k is the marginal probability of category k across both raters
    pi = {k: (a.count(k) + b.count(k)) / (2.0 * n) for k in categories}
    pe = (1.0 / (q - 1)) * sum(pi[k] * (1.0 - pi[k]) for k in categories)
    
    if math.isclose(pe, 1.0, abs_tol=1e-9):
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha(
    units: dict[Any, dict[str, Any]],
    level_of_measurement: Literal["nominal", "ordinal", "interval"] = "nominal"
) -> float:
    """Compute Krippendorff's Alpha (α) for reliability analysis.
    
    Parameters:
        units: dict mapping unit_id -> {rater_id: label/rating}
               Handles arbitrary numbers of raters and missing values per unit.
        level_of_measurement: 'nominal', 'ordinal', or 'interval'
    
    Returns:
        float: Krippendorff's alpha (1.0 = perfect agreement, 0.0 = chance agreement, <0 = systematic disagreement).
    """
    # Filter units with at least 2 rater annotations
    usable_units = {u: ratings for u, ratings in units.items() if len(ratings) >= 2}
    if not usable_units:
        return 1.0

    # Gather all observed values
    all_values = []
    for ratings in usable_units.values():
        all_values.extend(ratings.values())
    unique_values = sorted(set(all_values))
    if len(unique_values) <= 1:
        return 1.0

    # Total pair count across units
    total_pairs = sum(len(ratings) * (len(ratings) - 1) for ratings in usable_units.values())
    if total_pairs == 0:
        return 1.0

    # Distance function d(v1, v2)
    def dist_sq(v1: Any, v2: Any) -> float:
        if level_of_measurement == "nominal":
            return 0.0 if v1 == v2 else 1.0
        elif level_of_measurement == "ordinal":
            # Ordinal distance based on ranks
            i1 = unique_values.index(v1)
            i2 = unique_values.index(v2)
            if i1 == i2:
                return 0.0
            lo, hi = sorted((i1, i2))
            mid_sum = sum(all_values.count(unique_values[k]) for k in range(lo, hi + 1))
            return ((mid_sum - (all_values.count(v1) + all_values.count(v2)) / 2.0) ** 2)
        elif level_of_measurement == "interval":
            try:
                return float((float(v1) - float(v2)) ** 2)
            except (ValueError, TypeError):
                return 0.0 if v1 == v2 else 1.0
        return 0.0 if v1 == v2 else 1.0

    # Observed disagreement Do
    d_o_sum = 0.0
    for ratings in usable_units.values():
        vals = list(ratings.values())
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                d_o_sum += 2.0 * dist_sq(vals[i], vals[j]) / (m - 1)
    
    total_observations = sum(len(ratings) for ratings in usable_units.values())
    d_o = d_o_sum / total_observations

    # Expected disagreement De
    val_counts = Counter(all_values)
    d_e_sum = 0.0
    for v1 in unique_values:
        for v2 in unique_values:
            if v1 != v2:
                d_e_sum += val_counts[v1] * val_counts[v2] * dist_sq(v1, v2)
    
    d_e = d_e_sum / (total_observations * (total_observations - 1)) if total_observations > 1 else 0.0

    if math.isclose(d_e, 0.0, abs_tol=1e-9):
        return 1.0
    return 1.0 - (d_o / d_e)


def pairwise_agreement_matrix(
    by_rater: dict[str, dict[int, str]],
    min_overlap: int = 5,
    metric: Literal["kappa", "ac1"] = "kappa"
) -> list[dict[str, Any]]:
    """Compute pairwise agreement across all rater pairs having >= min_overlap tasks."""
    raters = sorted(by_rater.keys())
    results = []
    fn = cohens_kappa if metric == "kappa" else gwet_ac1
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            common = sorted(set(by_rater[r1].keys()) & set(by_rater[r2].keys()))
            if len(common) >= min_overlap:
                val = fn([by_rater[r1][t] for t in common], [by_rater[r2][t] for t in common])
                results.append({
                    "rater_a": r1,
                    "rater_b": r2,
                    "overlap": len(common),
                    "metric": metric,
                    "score": round(val, 4)
                })
    return results
