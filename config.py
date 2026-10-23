from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"
QUERY_DIR = DATA_DIR / "query"
QUERY_TXT_DIR = DATA_DIR / "query_txt"

OUTPUT_ROOT = ROOT / "outputs"
DEEP_OUT = OUTPUT_ROOT / "deep"
DEEP_FEATURES = DEEP_OUT / "features"
DEEP_CHUNKS = DEEP_FEATURES / "chunks"
DEEP_RANKLISTS = DEEP_OUT / "ranklists"
DEEP_FIGURES = DEEP_OUT / "figures"
DEEP_LOGS = DEEP_OUT / "logs"
DEEP_WEIGHTS = DEEP_OUT / "weights"
MANIFEST_PATH = DEEP_OUT / "manifest.json"

HAND_OUT = OUTPUT_ROOT / "hand"
HAND_FEATURES = HAND_OUT / "features"
HAND_CHUNKS = HAND_FEATURES / "sift"
HAND_RANKLISTS = HAND_OUT / "ranklists"
HAND_FIGURES = HAND_OUT / "figures"

NUM_GALLERY = 28_493
NUM_QUERIES = 50

# Hyperparameters of the ResNet
@dataclass(frozen=True)
class DeepCfg:
    backbone: str = "resnet50"
    weights: str = "IMAGENET1K_V2"
    image_size: int = 448
    long_side: int = 896
    query_short_side: int = 320
    query_long_side: int = 640
    query_pad: float = 0.15
    gem_p: float = 3.0
    pca_dim: int = 512
    pca_sample: int = 8192
    batch_size: int = 32
    top_k: int = 100
    window_scales: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8)
    window_aspect_mults: tuple[float, ...] = (0.7, 1.0, 1.4)
    window_stride: float = 0.35
    window_max: int = 160
    window_min_cells: int = 2
    aqe_topk: int = 5
    aqe_alpha: float = 0.5
    fusion_alpha: float = 0.35
    torch_threads: int = 16
    seed: int = 0

DEEP = DeepCfg()

# Hyperparameters of the SIFT
@dataclass(frozen=True)
class HandCfg:
    nfeatures: int = 400
    contrast_threshold: float = 0.01
    vocab_size: int = 4096
    vocab_sample: int = 400_000
    bow_topk: int = 200            # candidates that go through RANSAC
    ratio: float = 0.75            # Lowe's ratio test
    ransac_thresh: float = 5.0
    min_inliers: int = 12
    workers: int = 12
    seed: int = 0

HAND = HandCfg()

# Determine whether the current Torch instance supports CUDA
def resolve_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# Create all output directories
def ensure_dirs() -> None:
    for path in (
        OUTPUT_ROOT,
        DEEP_OUT,
        DEEP_FEATURES,
        DEEP_CHUNKS,
        DEEP_RANKLISTS,
        DEEP_FIGURES,
        DEEP_LOGS,
        DEEP_WEIGHTS,
        HAND_OUT,
        HAND_FEATURES,
        HAND_CHUNKS,
        HAND_RANKLISTS,
        HAND_FIGURES,
    ):
        path.mkdir(parents=True, exist_ok=True)
