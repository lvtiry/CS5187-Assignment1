# Deep learning
# 1) Feature extraction across the entire database and PCA whitening and fitting;
# 2) DeepRetriever: Load cached features, perform global recall + regional re-ranking + fusion ranking on the query;
# 3) retrieve_all: Iterate through 50 queries, generate a ranklist and a detailed file for visualization.

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEEP,
    DEEP_CHUNKS,
    DEEP_FEATURES,
    DEEP_RANKLISTS,
    MANIFEST_PATH,
    NUM_QUERIES,
    ensure_dirs,
    resolve_device,
)
from common.data import (
    build_manifest,
    crop_box,
    gallery_path,
    load_manifest,
    load_rgb,
    query_boxes,
    query_path,
)
from deepLearning.encoder import (
    Encoder,
    GalleryDataset,
    Whitening,
    target_size,
    to_tensor,
)
from deepLearning.rerank import integral_map, pool_windows, refine_box, window_boxes
from deepLearning.search import average_query_expansion, top_k

CHUNK = 1024

# Get the path to the feature file
def feature_paths(limit: int | None = None) -> dict[str, Path]:
    tag = f"_smoke{limit}" if limit else ""
    return {
        "ids": DEEP_FEATURES / f"gallery_ids{tag}.npy",
        "raw": DEEP_FEATURES / f"gallery_raw{tag}.npy",
        "pca": DEEP_FEATURES / f"gallery_pca512{tag}.npy",
        "whitening": DEEP_FEATURES / f"pca_whitening{tag}.npz",
    }

# Returns the chunk cache directory
def chunk_dir(limit: int | None = None) -> Path:
    return DEEP_CHUNKS / f"smoke{limit}" if limit else DEEP_CHUNKS

Box = tuple[int, int, int, int]

# Extract GeM descriptors from the entire database in chunks
def extract_gallery(limit: int | None = None, device: str | None = None,
                    verbose: bool = True) -> dict:
    ensure_dirs()
    device = device or resolve_device()
    manifest = load_manifest() if MANIFEST_PATH.exists() else build_manifest()
    meta = {g["id"]: (g["width"], g["height"]) for g in manifest["gallery"]}
    ids = sorted(meta)
    if limit:
        ids = ids[:limit]
    sizes = {gid: target_size(*meta[gid], DEEP.image_size, DEEP.long_side) for gid in ids}

    t0 = time.perf_counter()
    encoder = Encoder(DEEP.backbone, DEEP.weights, device)
    if verbose:
        print(f"encoder  : {DEEP.backbone} ({DEEP.weights}) dim={encoder.dim} on {device} "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"images   : {len(ids):,}  chunk={CHUNK}  batch={DEEP.batch_size}")

    starts = list(range(0, len(ids), CHUNK))
    chunks = chunk_dir(limit)
    chunks.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seen = 0
    for index, start in enumerate(starts, 1):
        chunk_ids = ids[start : start + CHUNK]
        path = chunks / f"chunk_{start:06d}.npz"
        seen += len(chunk_ids)
        if path.exists():
            if verbose:
                print(f"[{index}/{len(starts)}] chunk {start:>6} cached", flush=True)
            continue

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for gid in chunk_ids:
            groups[sizes[gid]].append(gid)
        position = {gid: i for i, gid in enumerate(chunk_ids)}
        feats = np.zeros((len(chunk_ids), encoder.dim), dtype=np.float32)

        for size, group in groups.items():
            loader = DataLoader(
                GalleryDataset(group, size),
                batch_size=DEEP.batch_size,
                num_workers=0,
                shuffle=False,
            )
            for batch_ids, batch in loader:
                desc = encoder.global_descriptors(batch)
                desc = desc.float().cpu().numpy()
                for gid, vec in zip(batch_ids.tolist(), desc):
                    feats[position[gid]] = vec

        np.savez(path, ids=np.asarray(chunk_ids, np.int32), feats=feats)
        elapsed = time.perf_counter() - started
        rate = seen / max(elapsed, 1e-6)
        eta = (len(ids) - seen) / max(rate, 1e-6)
        if verbose:
            print(f"[{index}/{len(starts)}] chunk {start:>6} {len(chunk_ids):>4} imgs "
                  f"| {rate:6.1f} img/s | ETA {eta / 60:4.1f} min", flush=True)

    if verbose:
        print(f"extraction finished in {(time.perf_counter() - started) / 60:.1f} min")
    return {"images": len(ids), "chunks": len(starts)}

# Merge all blocks
def merge_features(limit: int | None = None, verbose: bool = True) -> dict:
    files = sorted(chunk_dir(limit).glob("chunk_*.npz"))
    if not files:
        raise RuntimeError("no chunks found — run the extract stage first")

    ids_parts, feat_parts = [], []
    for path in files:
        data = np.load(path)
        ids_parts.append(data["ids"])
        feat_parts.append(data["feats"])
    ids = np.concatenate(ids_parts)
    feats = np.concatenate(feat_parts)

    order = np.argsort(ids)
    ids, feats = ids[order], feats[order]
    if len(ids) != len(np.unique(ids)):
        raise RuntimeError("duplicate ids across chunks")

    paths = feature_paths(limit)
    np.save(paths["ids"], ids)
    np.save(paths["raw"], feats)
    whitening = Whitening.fit(feats)
    whitening.save(paths["whitening"])
    reduced = whitening.transform(feats)
    np.save(paths["pca"], reduced)

    if verbose:
        print(f"merged   : {len(ids):,} descriptors, raw {feats.shape[1]}d -> "
              f"{reduced.shape[1]}d ({reduced.nbytes / 1e6:.0f} MB)")
    return {"images": len(ids), "raw_dim": int(feats.shape[1]), "pca_dim": int(reduced.shape[1])}

# Candidate result: ID, global score, regional score, fusion score, and prediction box
@dataclass
class Candidate:
    id: int
    global_score: float
    region_score: float
    fused: float
    box: Box

# retriever: loads cached features and the model, and supports query retrieval and candidate encoding
class DeepRetriever:
    # Load cached features, whitening parameters, and the model, and pre-compute the input dimensions for each image.
    def __init__(self, device: str | None = None, topk: int = DEEP.top_k):
        paths = feature_paths()
        for path in paths.values():
            if not path.exists():
                raise RuntimeError(f"{path.name} missing — run the extract stage first")
        self.device = device or resolve_device()
        self.topk = topk
        self.ids = np.load(paths["ids"]).astype(np.int64)
        self.feats = np.load(paths["pca"]).astype(np.float32)
        self.whitening = Whitening.load(paths["whitening"])
        self.encoder = Encoder(DEEP.backbone, DEEP.weights, self.device)

        manifest = load_manifest()
        self.image_sizes = {g["id"]: (g["width"], g["height"]) for g in manifest["gallery"]}
        self.sizes = {
            gid: target_size(w, h, DEEP.image_size, DEEP.long_side)
            for gid, (w, h) in self.image_sizes.items()
        }

    # Encodes the query fragment into a whitened descriptor
    def encode_patch(self, patch) -> np.ndarray:
        fmap, _, _, _ = self.encoder.encode_pil(patch, DEEP.query_short_side, DEEP.query_long_side)
        raw = self.encoder.descriptor(fmap).float().cpu().numpy()
        return self.whitening.transform(raw)

    # Retrieves results one by one based on the labels in the query box, then combines them into a complete ranking.
    def search(self, qid: int, topk: int | None = None, aqe: bool = True):
        image = load_rgb(query_path(qid))
        boxes = query_boxes(qid)
        descs = [self.encode_patch(crop_box(image, box, DEEP.query_pad)) for box in boxes]
        aspects = [box[2] / max(1.0, box[3]) for box in boxes]
        return self.search_descriptors(descs, aspects, topk=topk or self.topk, aqe=aqe)

    # Performs recall and reordering
    def search_descriptors(self, descs, aspects, topk: int | None = None, aqe: bool = True):
        topk = topk or self.topk
        best: dict[int, Candidate] = {}
        per_query: list[np.ndarray] = []

        for desc, aspect in zip(descs, aspects):
            sims = self.global_scores(desc, aqe=aqe)
            per_query.append(sims)
            rows = top_k(sims, topk)
            cand_ids = [int(self.ids[r]) for r in rows]
            region = self.region_scores(cand_ids, desc, aspect)

            g = np.asarray([sims[r] for r in rows], dtype=np.float32)
            r = np.asarray([region[gid][0] for gid in cand_ids], dtype=np.float32)
            # Raw cosine values stay comparable across query boxes; normalising
            # per box would let a weak but internally "best" candidate outrank a
            # strong one.
            fused = DEEP.fusion_alpha * g + (1.0 - DEEP.fusion_alpha) * r

            for j, gid in enumerate(cand_ids):
                previous = best.get(gid)
                if previous is None or fused[j] > previous.fused:
                    best[gid] = Candidate(gid, float(g[j]), float(r[j]), float(fused[j]),
                                          region[gid][1])

        score = np.max(np.stack(per_query), axis=0) if len(per_query) > 1 else per_query[0]
        head = sorted(best.values(), key=lambda c: -c.fused)
        head_ids = [c.id for c in head]
        seen = set(head_ids)
        tail = [int(self.ids[r]) for r in np.argsort(-score) if int(self.ids[r]) not in seen]
        return head_ids + tail, best, score

    # Calculates the cosine score between the query and the entire database
    def global_scores(self, desc: np.ndarray, aqe: bool = True) -> np.ndarray:
        query = desc
        if aqe and DEEP.aqe_topk > 0:
            query = average_query_expansion(desc, self.feats, DEEP.aqe_topk, DEEP.aqe_alpha)
        return self.feats @ query

    # Perform a window scan on each candidate, returning the similarity and pixel bounding box of the best window
    def region_scores(self, ids, query_desc: np.ndarray, aspect: float) -> dict:
        out: dict[int, tuple[float, Box]] = {}
        for gid, fmap in self.encode_ids(ids).items():
            integral = integral_map(fmap, DEEP.gem_p)
            hf, wf = fmap.shape[-2], fmap.shape[-1]
            boxes = window_boxes(hf, wf, base_aspect=aspect)
            pooled = pool_windows(integral, boxes, DEEP.gem_p).numpy()
            sims = self.whitening.transform(pooled) @ query_desc
            k = int(np.argmax(sims))
            width, height = self.image_sizes[gid]
            cell_box = refine_box(boxes, sims)
            out[gid] = (float(sims[k]), self.cell_box_to_pixels(cell_box, wf, hf, width, height))
        return out

    # Batch-code images in the image library with specified IDs
    def encode_ids(self, ids) -> dict[int, torch.Tensor]:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for gid in ids:
            groups[self.sizes[int(gid)]].append(int(gid))
        maps: dict[int, torch.Tensor] = {}
        for size, group in groups.items():
            for start in range(0, len(group), DEEP.batch_size):
                chunk = group[start : start + DEEP.batch_size]
                batch = torch.stack([to_tensor(load_rgb(gallery_path(g)), size) for g in chunk])
                for gid, fmap in zip(chunk, self.encoder.feature_maps(batch)):
                    maps[gid] = fmap.cpu()
        return maps

    # Converts feature map cell coordinates back to the original image pixel coordinates
    @staticmethod
    def cell_box_to_pixels(box: Box, wf: int, hf: int, width: int, height: int) -> Box:
        x0, y0, x1, y1 = box
        sx, sy = width / wf, height / hf
        return (
            int(round(x0 * sx)),
            int(round(y0 * sy)),
            int(round((x1 - x0) * sx)),
            int(round((y1 - y0) * sy)),
        )

# Iterate through all queries and write the ranklist
def retrieve_all(limit_queries: int | None = None, topk: int | None = None,
                 out_path: Path | None = None, device: str | None = None,
                 verbose: bool = True) -> dict:
    from common.metrics import validate_ranklist, write_ranklist

    ensure_dirs()
    out_path = Path(out_path) if out_path else DEEP_RANKLISTS / "run1.txt"
    engine = DeepRetriever(device=device, topk=topk or DEEP.top_k)

    qids = sorted(int(p.stem) for p in (query_path(0).parent).glob("*.jpg"))
    if limit_queries:
        qids = qids[:limit_queries]

    rankings: list[list[int]] = []
    details: dict[str, list[dict]] = {}
    started = time.perf_counter()

    for index, qid in enumerate(qids, 1):
        t0 = time.perf_counter()
        order, best, _ = engine.search(qid)
        rankings.append(order)

        top = sorted(best.values(), key=lambda c: -c.fused)[:30]
        details[str(qid)] = [
            {
                "id": c.id,
                "fused": round(c.fused, 5),
                "global": round(c.global_score, 5),
                "region": round(c.region_score, 5),
                "box": list(c.box),
            }
            for c in top
        ]
        if verbose:
            print(f"[{index:2d}/{len(qids)}] Q{qid:<2d} {time.perf_counter() - t0:5.1f}s "
                  f"top1={top[0].id:<6d} fused={top[0].fused:.3f} "
                  f"g={top[0].global_score:.3f} r={top[0].region_score:.3f}", flush=True)

    write_ranklist(out_path, rankings)
    detail_path = DEEP_RANKLISTS / "run1_detail.json"
    payload = {
        "method": "resnet50 + GeM + PCA whitening + regional re-ranking",
        "fusion_alpha": DEEP.fusion_alpha,
        "topk": DEEP.top_k,
        "queries": details,
    }
    detail_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    elapsed = time.perf_counter() - started
    errors = []
    if len(qids) == NUM_QUERIES:
        errors = validate_ranklist(out_path, len(engine.ids), NUM_QUERIES,
                                   valid_ids=engine.ids.tolist())
    if verbose:
        print(f"\nrank list -> {out_path}  ({len(qids)} rows, {elapsed / 60:.1f} min)")
        print(f"details   -> {detail_path}")
        print("validation: " + ("PASS" if not errors else f"FAIL ({len(errors)})"))
        for err in errors[:5]:
            print("  -", err)
    return {"path": str(out_path), "rows": len(qids), "errors": errors,
            "seconds": elapsed}
