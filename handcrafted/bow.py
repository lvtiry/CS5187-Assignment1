# Bag of words
from __future__ import annotations

import numpy as np

from config import HAND

# Training a visual vocabulary using k-means on sampled descriptors
def build_vocabulary(descriptors: np.ndarray, k: int = HAND.vocab_size,
                     seed: int = HAND.seed, verbose: bool = True) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    descriptors = np.asarray(descriptors, dtype=np.float32)
    k = int(min(k, len(descriptors)))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=10_000,
        n_init=3,
        max_iter=100,
        random_state=seed,
        verbose=0,
    )
    kmeans.fit(descriptors)
    if verbose:
        print(f"vocabulary: k={k} from {len(descriptors):,} descriptors")
    return kmeans.cluster_centers_.astype(np.float32)

# Assign each descriptor to the nearest visual word
def assign(descriptors: np.ndarray, centers: np.ndarray, block: int = 8192) -> np.ndarray:
    if len(descriptors) == 0:
        return np.empty(0, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float32)
    center_norm = (centers ** 2).sum(1)[None, :]
    out = np.empty(len(descriptors), dtype=np.int32)
    for start in range(0, len(descriptors), block):
        chunk = descriptors[start : start + block]
        dist = center_norm - 2.0 * (chunk @ centers.T)
        out[start : start + block] = dist.argmin(1)
    return out

# Generate a word frequency histogram for a chart
def histogram(indices: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(indices, minlength=k).astype(np.float32)

# Multiply word frequency by IDF, then perform L2 normalization to obtain a vector for retrieval.
def tfidf_vector(hist: np.ndarray, idf: np.ndarray) -> np.ndarray:
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    vector = (hist / total) * idf
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

# Get IDF
def build_idf(histograms, num_documents: int) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(histograms):
        df = np.asarray((histograms > 0).sum(0)).ravel().astype(np.float32)
    else:
        df = np.zeros(histograms.shape[1], dtype=np.float32)
        for start in range(0, histograms.shape[0], 1024):
            df += (histograms[start : start + 1024] > 0).sum(0).astype(np.float32)
    return np.log((1.0 + num_documents) / (1.0 + df)).astype(np.float32) + 1.0
