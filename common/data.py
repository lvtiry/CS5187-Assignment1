# Responsible for enumerating image library/query IDs,
# Parsing annotation boxes (x, y, w, h),
# Reading, scaling, and cropping images,
# Scanning the entire dataset to generate a manifest
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    GALLERY_DIR,
    MANIFEST_PATH,
    NUM_GALLERY,
    NUM_QUERIES,
    QUERY_DIR,
    QUERY_TXT_DIR,
    ensure_dirs,
)

# Bounding box, top-left coordinates, width, height
Box = tuple[int, int, int, int]

# Get bounding box
def parse_bbox_file(path: str | Path) -> list[Box]:
    path = Path(path)
    boxes: list[Box] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"{path.name}:{lineno}: expected 4 numbers, got {len(parts)}: {line!r}")
        x, y, w, h = (int(round(float(v))) for v in parts)
        boxes.append((x, y, w, h))
    if not boxes:
        raise ValueError(f"{path.name}: no bounding box found")
    return boxes

# Convert (x, y, w, h) to (x1, y1, x2, y2)
def box_to_xyxy(box: Box) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h

# Crop the frame to fit within the image
def clip_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

# Return the ID numbers of all images in the gallery
def gallery_ids() -> list[int]:
    return sorted(int(p.stem) for p in GALLERY_DIR.glob("*.jpg"))

# Return the ID of the entire query
def query_ids() -> list[int]:
    return sorted(int(p.stem) for p in QUERY_DIR.glob("*.jpg"))

# Construct the image path from the image library ID
def gallery_path(gid: int) -> Path:
    return GALLERY_DIR / f"{gid}.jpg"

# Get the image path
def query_path(qid: int) -> Path:
    return QUERY_DIR / f"{qid}.jpg"

# Construct the image path from the query ID
def query_boxes(qid: int) -> list[Box]:
    return parse_bbox_file(QUERY_TXT_DIR / f"{qid}.txt")

# Get the image dimensions
def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size

# Scale proportionally along the shorter side
def resize_short_side(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale = target / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

# Optional scaling based on the shorter side
def load_rgb(path: str | Path, short_side: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return resize_short_side(img, short_side) if short_side else img

# Crop Image
def crop_box(img: Image.Image, box: Box, pad: float = 0.0) -> Image.Image:
    w, h = img.size
    x, y, bw, bh = clip_box(box, w, h)
    if pad:
        cx, cy = x + bw / 2.0, y + bh / 2.0
        bw, bh = bw * (1.0 + pad), bh * (1.0 + pad)
        x, y = cx - bw / 2.0, cy - bh / 2.0
    left, top = max(0, int(round(x))), max(0, int(round(y)))
    right, bottom = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    return img.crop((left, top, max(left + 1, right), max(top + 1, bottom)))

# Scans two sets of data and generates a manifest containing dimensions and annotation boxes
def build_manifest(save: bool = True) -> dict:
    gallery = []
    for gid in gallery_ids():
        width, height = image_size(gallery_path(gid))
        gallery.append({"id": gid, "width": width, "height": height})

    queries = []
    for qid in query_ids():
        width, height = image_size(query_path(qid))
        boxes = query_boxes(qid)
        queries.append(
            {
                "id": qid,
                "width": width,
                "height": height,
                "boxes": [list(b) for b in boxes],
            }
        )

    manifest = {
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "gallery": gallery,
        "queries": queries,
    }
    if save:
        ensure_dirs()
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest

def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

# Generate manifest
def scan(verbose: bool = True) -> dict:
    # build_manifest() is responsible for iterating through the images
    manifest = build_manifest(save=True)
    gallery, queries = manifest["gallery"], manifest["queries"]
    problems: list[str] = []

    # Check whether the images in the gallery have been read correctly
    # Check whether the number of images is correct and whether the image IDs are consecutive
    # Non-consecutive IDs indicate a problem with the reading process
    ids = [g["id"] for g in gallery]
    contiguous = bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))
    if len(gallery) != NUM_GALLERY:
        problems.append(f"gallery count {len(gallery)} != {NUM_GALLERY}")
    if not contiguous:
        problems.append("gallery ids are not contiguous")

    # Check whether the images in the query list were loaded correctly
    if len(queries) != NUM_QUERIES:
        problems.append(f"query count {len(queries)} != {NUM_QUERIES}")

    # Check each query for duplicates
    missing = [q["id"] for q in queries if not (QUERY_TXT_DIR / f"{q['id']}.txt").exists()]
    if missing:
        problems.append(f"{len(missing)} queries without annotation file")

    # Retrieve information about the “box” from the query
    boxes_per_query: dict[int, int] = {}
    areas, ratios, outside = [], [], 0
    for q in queries:
        boxes_per_query[len(q["boxes"])] = boxes_per_query.get(len(q["boxes"]), 0) + 1
        for x, y, w, h in q["boxes"]:
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > q["width"] or y + h > q["height"]:
                outside += 1
            areas.append((w * h) / float(q["width"] * q["height"]))
            ratios.append(w / float(h))
    if outside:
        problems.append(f"{outside} boxes exceed the image boundary")

    # Sort by Image Size
    sizes: dict[tuple[int, int], int] = {}
    for g in gallery:
        key = (g["width"], g["height"])
        sizes[key] = sizes.get(key, 0) + 1
    top_sizes = sorted(sizes.items(), key=lambda kv: -kv[1])[:3]

    # Calculate and Display
    stats = {
        "num_gallery": len(gallery),
        "num_queries": len(queries),
        "id_range": (min(ids), max(ids)) if ids else (0, 0),
        "contiguous": contiguous,
        "missing_annotations": len(missing),
        "boxes_per_query": dict(sorted(boxes_per_query.items())),
        "box_area_ratio": (
            float(np.min(areas)), float(np.median(areas)), float(np.max(areas))
        ),
        "box_aspect": (float(np.min(ratios)), float(np.median(ratios)), float(np.max(ratios))),
        "boxes_outside": outside,
        "top_sizes": top_sizes,
        "manifest": str(MANIFEST_PATH),
    }

    if verbose:
        lo, hi = stats["id_range"]
        print(f"gallery images : {len(gallery):,} (expected {NUM_GALLERY:,})")
        print(f"id range       : {lo} .. {hi}  contiguous={contiguous}")
        print("top image sizes: " + ", ".join(f"{w}x{h}:{n}" for (w, h), n in top_sizes))
        print(f"queries        : {len(queries)} (expected {NUM_QUERIES}), missing .txt={len(missing)}")
        print(f"instances/query: {stats['boxes_per_query']}")
        a = stats["box_area_ratio"]
        r = stats["box_aspect"]
        print(f"box area ratio : min={a[0]:.3f} median={a[1]:.3f} max={a[2]:.3f}")
        print(f"box aspect w/h : min={r[0]:.2f} median={r[1]:.2f} max={r[2]:.2f}")
        print(f"boxes outside  : {outside}")
        print(f"manifest -> {MANIFEST_PATH}")

    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        raise RuntimeError("dataset integrity check failed: " + "; ".join(problems))
    if verbose:
        print("dataset OK")
    return stats
