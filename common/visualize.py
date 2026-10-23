# Generate figures
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from config import DEEP_FIGURES, DEEP_RANKLISTS, GALLERY_DIR, QUERY_DIR, ensure_dirs
from common.data import crop_box, query_boxes

GT_COLOR = "#2ecc71"
PRED_COLOR = "#f1c40f"

# Read RGB data into a NumPy array
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))

# Draw a set of boxes on the coordinate axes
def draw(ax, boxes, color, linewidth=2.4):
    for x, y, w, h in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth))

# Crops the individual annotation boxes from the query and combines them horizontally into a single image
def query_montage(qid: int, height: int = 230, gap: int = 12) -> Image.Image:
    """The cropped instance(s) the encoder actually receives, side by side."""
    img = Image.open(QUERY_DIR / f"{qid}.jpg").convert("RGB")
    crops = [crop_box(img, box, 0.15) for box in query_boxes(qid)]
    scaled = [
        c.resize((max(1, int(round(c.width * height / c.height))), height), Image.BICUBIC)
        for c in crops
    ]
    width = sum(s.width for s in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (max(1, width), height), "white")
    x = 0
    for s in scaled:
        canvas.paste(s, (x, 0))
        x += s.width + gap
    return canvas

# Generate a top-k box plot for each of the first 5 queries and return a list of the generated files.
def render(queries: int = 5, num_results: int = 10, detail_path: Path | None = None,
           out_dir: Path | None = None, verbose: bool = True) -> list[Path]:
    ensure_dirs()
    detail_path = Path(detail_path) if detail_path else DEEP_RANKLISTS / "run1_detail.json"
    out_dir = Path(out_dir) if out_dir else DEEP_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(detail_path.read_text(encoding="utf-8"))
    details = payload["queries"]
    written: list[Path] = []

    for qid in range(queries):
        hits = details[str(qid)][:num_results]
        fig, axes = plt.subplots(2, 6, figsize=(11.8, 4.7))
        for ax in axes.ravel():
            ax.axis("off")

        ax = axes[0, 0]
        ax.imshow(load_rgb(QUERY_DIR / f"{qid}.jpg"))
        draw(ax, query_boxes(qid), GT_COLOR)
        ax.set_title(f"Query Q{qid + 1}", fontsize=9.5, fontweight="bold")
        ax.axis("off")

        ax = axes[1, 0]
        ax.imshow(np.asarray(query_montage(qid)))
        ax.set_title("query crop(s)", fontsize=8.5)
        ax.axis("off")

        for rank, hit in enumerate(hits, 1):
            row = 0 if rank <= 5 else 1
            col = rank if rank <= 5 else rank - 5
            ax = axes[row, col]
            ax.imshow(load_rgb(GALLERY_DIR / f"{hit['id']}.jpg"))
            if hit.get("box"):
                draw(ax, [hit["box"]], PRED_COLOR)
            if "fused" in hit:
                label = (
                    f"#{rank}  id={hit['id']}  s={hit['fused']:.2f}\n"
                    f"g={hit['global']:.2f} r={hit['region']:.2f}"
                )
            else:
                label = (
                    f"#{rank}  id={hit['id']}  bow={hit.get('bow', 0):.2f}\n"
                    f"inliers={hit.get('inliers', 0)}"
                )
            ax.set_title(label, fontsize=8.5)
            ax.axis("off")

        fig.tight_layout()
        out = out_dir / f"q{qid + 1:02d}_top{num_results}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        if verbose:
            print(f"figure -> {out}")

    return written
