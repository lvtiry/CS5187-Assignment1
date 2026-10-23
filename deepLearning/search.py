# Vector search
from __future__ import annotations

import numpy as np

from deepLearning.encoder import l2_normalize

# Returns the k indices with the highest scores
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, scores.shape[0]))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


# Average Query Extension
def average_query_expansion(
    query: np.ndarray,
    matrix: np.ndarray,
    topk: int = 5,
    alpha: float = 0.5,
    beta: float = 3.0,
) -> np.ndarray:

    if topk <= 0:
        return query
    sims = matrix @ query
    idx = top_k(sims, topk)
    weights = np.clip(sims[idx], 0.0, None) ** beta
    if weights.sum() <= 0:
        return query
    expanded = (weights[:, None] * matrix[idx]).sum(0) / weights.sum()
    return l2_normalize(query + alpha * expanded)
