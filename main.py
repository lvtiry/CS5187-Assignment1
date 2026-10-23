from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DEEP,
    DEEP_RANKLISTS,
    HAND,
    HAND_FIGURES,
    HAND_RANKLISTS,
    ensure_dirs,
    resolve_device,
)

DEEP_STAGES = ("scan", "extract", "retrieve", "validate", "visualize")
HAND_STAGES = ("scan", "extract", "index", "retrieve", "visualize")


# Parse the value of --stage into a list of stages
def parse_stages(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    stages = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [s for s in stages if s not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {unknown}; choose from {allowed} or 'all'"
        )
    return stages

# Deep learning
# Scan the dataset and generate a manifest
def stage_scan(args) -> None:
    from common.data import scan

    scan(verbose=True)

# Extract deep features from the entire database and perform PCA whitening
def stage_extract(args) -> None:
    from deepLearning.pipeline import extract_gallery, merge_features

    info = extract_gallery(limit=args.limit, device=args.device, verbose=True)
    merge_features(limit=args.limit, verbose=True)
    print(f"features ready: {info['images']:,} images")

# Retrieve all queries and write them to run1.txt
def stage_retrieve(args) -> None:
    from deepLearning.pipeline import retrieve_all

    result = retrieve_all(limit_queries=args.limit_queries, device=args.device, verbose=True)
    if result["errors"]:
        raise SystemExit(1)

# Verify run1.txt
def stage_validate(args) -> None:
    from common.data import gallery_ids
    from common.metrics import validate_ranklist

    path = DEEP_RANKLISTS / "run1.txt"
    ids = gallery_ids()
    errors = validate_ranklist(path, len(ids), 50, valid_ids=ids)
    if errors:
        print(f"FAIL {path} ({len(errors)} problem(s))")
        for err in errors[:10]:
            print("  -", err)
        raise SystemExit(1)
    print(f"PASS {path} (50 rows x {len(ids):,} ids)")

# Top 10 Rendering Images
def stage_visualize(args) -> None:
    from common.visualize import render

    render(queries=args.figures, num_results=10, verbose=True)

# Executes each phase in sequence and measures the time.
def run_deep(args) -> int:
    stages = parse_stages(args.stage, DEEP_STAGES)
    device = args.device or resolve_device()
    print(f"method   : deep ({DEEP.backbone}, {DEEP.weights})")
    print(f"device   : {device}")
    print(f"gallery  : {DEEP.image_size}px short side, top-k={DEEP.top_k}, PCA={DEEP.pca_dim}d")
    print(f"stages   : {', '.join(stages)}")

    ensure_dirs()
    started = time.perf_counter()
    handlers = {
        "scan": stage_scan,
        "extract": stage_extract,
        "retrieve": stage_retrieve,
        "validate": stage_validate,
        "visualize": stage_visualize,
    }
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        t0 = time.perf_counter()
        handlers[stage](args)
        print(f"--- {stage} finished in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    return 0

# Handcrafted SIFT
# Extracting SIFT descriptors from the entire database using multiple processes
def hand_extract(args) -> None:
    from handcrafted.pipeline import extract_all

    extract_all(limit=args.limit, workers=args.workers, verbose=True)

# Train the vocabulary and build a BoW sparse index
def hand_index(args) -> None:
    from handcrafted.pipeline import build_index

    build_index(verbose=True)

# BoW recall + RANSAC validation, write run2.txt
def hand_retrieve(args) -> None:
    from handcrafted.pipeline import retrieve_all

    result = retrieve_all(limit_queries=args.limit_queries, verbose=True)
    if result["errors"]:
        raise SystemExit(1)

# Top 10 Rendering Images
def hand_visualize(args) -> None:
    from common.visualize import render

    render(queries=args.figures, num_results=10,
           detail_path=HAND_RANKLISTS / "run2_detail.json",
           out_dir=HAND_FIGURES, verbose=True)


# Executes each phase in sequence and measures the time.
def run_hand(args) -> int:
    stages = parse_stages(args.stage, HAND_STAGES)
    print(f"method   : hand (RootSIFT + BoW + RANSAC)")
    print(f"settings : nfeatures={HAND.nfeatures}, vocab={HAND.vocab_size}, "
          f"topk={HAND.bow_topk}, workers={args.workers}")
    print(f"stages   : {', '.join(stages)}")

    ensure_dirs()
    started = time.perf_counter()
    handlers = {
        "scan": stage_scan,
        "extract": hand_extract,
        "index": hand_index,
        "retrieve": hand_retrieve,
        "visualize": hand_visualize,
    }
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        t0 = time.perf_counter()
        handlers[stage](args)
        print(f"--- {stage} finished in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    return 0


# Building a Command-Line Parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="method", required=True)

    deep = sub.add_parser("deep", help="ResNet-50 pipeline (method A)")
    deep.add_argument("--stage", default="all",
                      help="comma-separated stages, or 'all': " + " | ".join(DEEP_STAGES))
    deep.add_argument("--limit", type=int, default=None, help="only the first N gallery images")
    deep.add_argument("--limit-queries", type=int, default=None, help="only the first N queries")
    deep.add_argument("--figures", type=int, default=5, help="how many query figures to render")
    deep.add_argument("--device", default=None, choices=["cpu", "cuda"])

    hand = sub.add_parser("hand", help="RootSIFT + BoW pipeline (method B)")
    hand.add_argument("--stage", default="all",
                      help="comma-separated stages, or 'all': " + " | ".join(HAND_STAGES))
    hand.add_argument("--limit", type=int, default=None, help="only the first N gallery images")
    hand.add_argument("--limit-queries", type=int, default=None, help="only the first N queries")
    hand.add_argument("--figures", type=int, default=5, help="how many query figures to render")
    hand.add_argument("--workers", type=int, default=HAND.workers, help="SIFT worker processes")
    hand.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.method == "deep":
        return run_deep(args)
    if args.method == "hand":
        return run_hand(args)
    raise SystemExit(f"unknown method {args.method!r}")


if __name__ == "__main__":
    raise SystemExit(main())

# .\.venv\Scripts\python.exe main.py deep --stage all
# .\.venv\Scripts\python.exe main.py hand --stage all