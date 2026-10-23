# Hand crafte Method
# 1) Use multiple processes to extract SIFT features from the entire database and save them to disk in blocks;
# 2) Train the vocabulary, encode all images, calculate TF-IDF, and save the sparse matrix;
# 3) Perform RANSAC validation and generate a ranklist.
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

from config import (
    HAND,
    HAND_CHUNKS,
    HAND_FEATURES,
    HAND_RANKLISTS,
    NUM_QUERIES,
    QUERY_DIR,
    QUERY_TXT_DIR,
    ensure_dirs,
)
from common.data import gallery_ids, query_boxes
from handcrafted.bow import assign, build_idf, build_vocabulary, histogram, tfidf_vector
from handcrafted.geometric import verify
from handcrafted.sift import extract_array, extract_gallery_image, imread_gray

CHUNK = 1024
VOCAB_PATH = HAND_FEATURES / "vocab.npz"
BOW_PATH = HAND_FEATURES / "bow.npz"
INDEX_PATH = HAND_FEATURES / "sift_index.npz"
Box = tuple[int, int, int, int]


# The directory corresponding to a particular block
def chunk_dir(start: int, limit: int | None = None) -> Path:
    base = HAND_CHUNKS / f"smoke{limit}" if limit else HAND_CHUNKS
    return base / f"chunk_{start:06d}"


# Multi-process Extraction of the Full RootSIFT Database
def extract_all(limit: int | None = None, workers: int = HAND.workers,
                verbose: bool = True) -> dict:
    # Initialize the output content
    ensure_dirs()
    ids = gallery_ids()

    # Check if there is a limit on the number of images that can be uploaded
    if limit:
        ids = ids[:limit]

    # Group into chunks based on the chunk annotations
    starts = list(range(0, len(ids), CHUNK))
    if verbose:
        print(f"sift     : {len(ids):,} images, chunk={CHUNK}, workers={workers}")

    # Create a process pool to run multiple processes simultaneously
    started = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        for index, start in enumerate(starts, 1):

            # Get the block number
            chunk_ids = ids[start : start + CHUNK]

            # Specify the output directory
            out = chunk_dir(start, limit)

            if (out / "ids.npy").exists():
                if verbose:
                    print(f"[{index}/{len(starts)}] chunk {start:>6} cached", flush=True)
                continue

            t0 = time.perf_counter()
            results = list(pool.map(extract_gallery_image, chunk_ids, chunksize=8))
            results.sort(key=lambda item: item[0])

            points = [r[1] for r in results]
            descs = [r[2] for r in results]
            lengths = np.array([len(d) for d in descs], dtype=np.int64)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "ids.npy", np.asarray([r[0] for r in results], np.int32))
            np.save(out / "offsets.npy", offsets)
            np.save(out / "kp.npy", np.concatenate(points).astype(np.float32))
            np.save(out / "desc.npy", np.concatenate(descs).astype(np.float32))

            elapsed = time.perf_counter() - started
            done = index * CHUNK
            rate = done / max(elapsed, 1e-6)
            if verbose:
                print(f"[{index}/{len(starts)}] chunk {start:>6} "
                      f"{lengths.sum():>7} descriptors | {rate:6.1f} img/s | "
                      f"{time.perf_counter() - t0:5.1f}s", flush=True)
    finally:
        pool.shutdown()

    if verbose:
        print(f"sift finished in {(time.perf_counter() - started) / 60:.1f} min")
    return {"images": len(ids), "chunks": len(starts)}


# List the directory of completed SIFT blocks
def _chunk_files() -> list[Path]:
    return sorted(p for p in HAND_CHUNKS.glob("chunk_*") if (p / "ids.npy").exists())


# Train the vocabulary, encode all images, calculate TF-IDF, and save the sparse matrix and its mapping to IDs
def build_index(limit: int | None = None, verbose: bool = True) -> dict:
    ensure_dirs()
    files = _chunk_files()
    if not files:
        raise RuntimeError("no SIFT chunks — run the extract stage first")

    # If the index already exists and the parameters match, reuse it directly
    if VOCAB_PATH.exists() and BOW_PATH.exists():
        try:
            cached = np.load(VOCAB_PATH)
            same = (
                int(cached["nfeatures"]) == HAND.nfeatures
                and int(cached["vocab_size"]) == len(cached["centers"])
                and int(cached["num_images"]) == len(np.load(HAND_FEATURES / "bow_ids.npy"))
            )
        except (KeyError, OSError):
            same = True
            if verbose:
                print("index: Parameter fingerprints are missing; reuse those from the existing file.")
        if same:
            if verbose:
                print(f"index already exists(nfeatures={HAND.nfeatures}, vocab={int(cached['vocab_size'])}, "
                      f"images={int(cached['num_images'])})")
            return {"images": int(cached["num_images"]), "vocab": int(cached["vocab_size"]),
                    "skipped": True}

    # Sample descriptors for k-means
    rng = np.random.default_rng(HAND.seed)
    per_chunk = max(1, HAND.vocab_sample // len(files))
    samples = []
    for path in files:
        desc = np.load(path / "desc.npy", mmap_mode="r")
        if len(desc) == 0:
            continue
        take = min(per_chunk, len(desc))
        idx = rng.choice(len(desc), size=take, replace=False)
        idx.sort()
        samples.append(np.asarray(desc[idx], dtype=np.float32))
    sample = np.concatenate(samples)
    centers = build_vocabulary(sample, HAND.vocab_size, HAND.seed, verbose)
    np.savez(VOCAB_PATH, centers=centers)

    # Encode, keeping only non-zero words (each image has <= nfeatures)
    rows, cols, vals, ids_all = [], [], [], []
    row_offset = 0
    for path in files:
        ids = np.load(path / "ids.npy")
        offsets = np.load(path / "offsets.npy")
        desc = np.load(path / "desc.npy", mmap_mode="r")
        for i, gid in enumerate(ids):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if end <= start:
                hist = np.zeros(len(centers), np.float32)
            else:
                words = assign(np.asarray(desc[start:end], dtype=np.float32), centers)
                hist = histogram(words, len(centers))
            nz = np.nonzero(hist)[0]
            rows.append(np.full(len(nz), row_offset, np.int32))
            cols.append(nz.astype(np.int32))
            vals.append(hist[nz] / max(hist.sum(), 1.0))
            ids_all.append(gid)
            row_offset += 1
        if verbose:
            print(f"encoded {row_offset:,} images", flush=True)

    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row_offset, len(centers)),
        dtype=np.float32,
    )
    idf = build_idf(matrix, row_offset)
    matrix = matrix.multiply(idf[None, :]).tocsr()
    norms = np.sqrt(matrix.multiply(matrix).sum(1)).A.ravel()
    norms[norms == 0] = 1.0
    matrix = sparse.diags(1.0 / norms) @ matrix

    ids_array = np.asarray(ids_all, np.int32)
    sparse.save_npz(BOW_PATH, matrix.tocsr())
    np.save(HAND_FEATURES / "bow_ids.npy", ids_array)
    np.save(HAND_FEATURES / "idf.npy", idf)

    np.savez(VOCAB_PATH, centers=centers, nfeatures=HAND.nfeatures,
             vocab_size=len(centers), num_images=row_offset)

    # id -> (chunk, row) lookup for verification
    lookup = {}
    for path in files:
        chunk_ids = np.load(path / "ids.npy")
        for i, gid in enumerate(chunk_ids):
            lookup[int(gid)] = (path.name, int(i))
    np.savez(INDEX_PATH, keys=np.array(list(lookup), np.int32),
             chunks=np.array([lookup[k][0] for k in lookup]),
             rows=np.array([lookup[k][1] for k in lookup], np.int32))

    if verbose:
        print(f"bow      : {matrix.shape[0]:,} x {matrix.shape[1]:,}, "
              f"{matrix.nnz:,} non-zeros")
    return {"images": matrix.shape[0], "vocab": matrix.shape[1], "nnz": int(matrix.nnz)}


# Create a retriever: Load the vocabulary/BoW index, perform recall and geometric validation
class BowRetriever:
    # Loading the vocabulary, the BoW sparse matrix, IDF, and the chunked index
    def __init__(self):
        if not BOW_PATH.exists():
            raise RuntimeError("bow.npz missing — run the index stage first")
        centers = np.load(VOCAB_PATH)["centers"]
        self.centers = centers
        self.matrix = sparse.load_npz(BOW_PATH).tocsr()
        self.ids = np.load(HAND_FEATURES / "bow_ids.npy").astype(np.int64)
        self.idf = np.load(HAND_FEATURES / "idf.npy")
        index = np.load(INDEX_PATH)
        self.lookup = {
            int(k): (str(c), int(r))
            for k, c, r in zip(index["keys"], index["chunks"], index["rows"])
        }
        self._cache: dict[str, dict] = {}

    # Loads and caches the descriptors, keypoints, and offset tables for a specific block on demand.
    def _chunk(self, name: str) -> dict:
        if name not in self._cache:
            path = HAND_CHUNKS / name
            self._cache[name] = {
                "desc": np.load(path / "desc.npy", mmap_mode="r"),
                "kp": np.load(path / "kp.npy", mmap_mode="r"),
                "offsets": np.load(path / "offsets.npy"),
            }
        return self._cache[name]

    # Runs RootSIFT on each bounding box in the query and returns coordinates, descriptor, bounding box.
    def encode_query(self, qid: int) -> list[tuple[np.ndarray, np.ndarray, Box]]:
        """RootSIFT inside each annotated box of the query image."""
        gray = imread_gray(QUERY_DIR / f"{qid}.jpg")
        if gray is None:
            raise FileNotFoundError(QUERY_DIR / f"{qid}.jpg")
        out = []
        for box in query_boxes(qid):
            x, y, w, h = box
            mask = np.zeros(gray.shape[:2], np.uint8)
            mask[max(0, y) : y + h, max(0, x) : x + w] = 255
            points, desc = extract_array(gray, mask)
            out.append((points, desc, box))
        return out

    # BoW retrieves the top-k results, verifies them one by one using RANSAC, and returns the sorted full list along with the candidate details.
    def search(self, qid: int, topk: int = HAND.bow_topk):
        queries = self.encode_query(qid)          # SIFT runs once per box
        scores = []
        for points, desc, box in queries:
            if len(desc) == 0:
                scores.append(np.zeros(len(self.ids), np.float32))
                continue
            words = assign(desc, self.centers)
            hist = histogram(words, len(self.centers))
            vector = tfidf_vector(hist, self.idf)
            scores.append(self.matrix @ vector)
        bow = np.max(np.stack(scores), axis=0)

        order = np.argsort(-bow)[:topk]
        verified, others = [], []
        for rank, row in enumerate(order):
            gid = int(self.ids[row])
            best_inliers, best_box = 0, None
            location = self.lookup.get(gid)
            for points, desc, qbox in queries:
                if location is None or len(desc) == 0:
                    continue
                chunk = self._chunk(location[0])
                i = location[1]
                start, end = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
                if end <= start:
                    continue
                inliers, box = verify(
                    points, desc,
                    np.asarray(chunk["kp"][start:end]),
                    np.asarray(chunk["desc"][start:end]),
                    qbox,
                )
                if inliers > best_inliers:
                    best_inliers, best_box = inliers, box
            entry = {
                "id": gid,
                "bow": float(bow[row]),
                "inliers": int(best_inliers),
                "box": list(best_box) if best_box else None,
                "rank": rank,
            }
            (verified if best_inliers >= HAND.min_inliers else others).append(entry)

        verified.sort(key=lambda e: -e["inliers"])
        head = verified + others
        head_ids = [e["id"] for e in head]
        seen = set(head_ids)
        tail = [int(self.ids[r]) for r in np.argsort(-bow) if int(self.ids[r]) not in seen]
        return head_ids + tail, head, bow


# Iterate through all queries, write to run2.txt and the detailed JSON file, and perform format validation.
def retrieve_all(limit_queries: int | None = None, topk: int = HAND.bow_topk,
                 verbose: bool = True) -> dict:
    from common.metrics import validate_ranklist, write_ranklist

    ensure_dirs()
    engine = BowRetriever()
    qids = sorted(int(p.stem) for p in QUERY_DIR.glob("*.jpg"))
    if limit_queries:
        qids = qids[:limit_queries]

    rankings, details = [], {}
    started = time.perf_counter()
    for index, qid in enumerate(qids, 1):
        t0 = time.perf_counter()
        order, head, _ = engine.search(qid, topk)
        rankings.append(order)
        details[str(qid)] = head[:30]
        if verbose:
            best = head[0]
            print(f"[{index:2d}/{len(qids)}] Q{qid:<2d} {time.perf_counter() - t0:5.1f}s "
                  f"top1={best['id']:<6d} bow={best['bow']:.3f} inliers={best['inliers']}",
                  flush=True)

    out_path = HAND_RANKLISTS / "run2.txt"
    write_ranklist(out_path, rankings)
    (HAND_RANKLISTS / "run2_detail.json").write_text(
        json.dumps({"method": "RootSIFT + BoW(tf-idf) + RANSAC", "queries": details}, indent=1),
        encoding="utf-8",
    )
    errors = []
    if len(qids) == NUM_QUERIES:
        errors = validate_ranklist(out_path, len(engine.ids), NUM_QUERIES,
                                   valid_ids=engine.ids.tolist())
    if verbose:
        print(f"\nrank list -> {out_path} ({(time.perf_counter() - started) / 60:.1f} min)")
        print("validation: " + ("PASS" if not errors else f"FAIL ({len(errors)})"))
    return {"path": str(out_path), "rows": len(qids), "errors": errors}
