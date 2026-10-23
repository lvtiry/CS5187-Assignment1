# Retrieval metrics and submission-format
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_ROW_RE = re.compile(r"^[Qq]\s*(\d+)\s*:\s*(.*)$")
_SEP_RE = re.compile(r"[,\s]+")

# Percentage of relevant results among the first k results
def precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant = set(relevant)
    return sum(1 for doc in list(ranked)[:k] if doc in relevant) / float(k)

#  Average Precision over the first k
def average_precision(ranked: Sequence[int], relevant: Iterable[int], k: int | None = None) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    ranked = list(ranked)
    if k is not None:
        ranked = ranked[:k]
    hits, total = 0, 0.0
    for rank, doc in enumerate(ranked, 1):
        if doc in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), len(ranked)) if ranked else 0.0

# Mean AP across multiple queries
def mean_average_precision(
    ranked_lists: Sequence[Sequence[int]],
    relevant_sets: Sequence[Iterable[int]],
    k: int | None = None,
) -> float:
    scores = [average_precision(r, s, k) for r, s in zip(ranked_lists, relevant_sets)]
    return float(np.mean(scores)) if scores else 0.0

# Write ranklist
def write_ranklist(path: str | Path, ranked_lists: Sequence[Sequence[int]], start: int = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for offset, ids in enumerate(ranked_lists):
            fh.write(f"Q{start + offset}: " + " ".join(str(int(d)) for d in ids) + "\n")
    return path

# Verify the ranklist file
def validate_ranklist(
    path: str | Path,
    num_gallery: int,
    num_queries: int,
    valid_ids: Sequence[int] | None = None,
) -> list[str]:
    path = Path(path)
    if not path.exists():
        return [f"{path}: file not found"]
    return validate_ranklist_text(
        path.read_text(encoding="utf-8"), num_gallery, num_queries, valid_ids
    )


# Validates the ranklist text in memory
def validate_ranklist_text(
    text: str,
    num_gallery: int,
    num_queries: int,
    valid_ids: Sequence[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    rows = [line.strip() for line in text.splitlines()]
    rows = [line for line in rows if line]
    if len(rows) != num_queries:
        errors.append(f"expected {num_queries} rows, found {len(rows)}")

    allowed = set(valid_ids) if valid_ids is not None else None
    for index, line in enumerate(rows):
        expected = index + 1
        match = _ROW_RE.match(line)
        if match is None:
            errors.append(f"row {expected}: missing 'Q{expected}:' prefix -> {line[:40]!r}")
            continue
        qid = int(match.group(1))
        if qid != expected:
            errors.append(f"row {expected}: expected 'Q{expected}:', found 'Q{qid}:'")
        tokens = [t for t in _SEP_RE.split(match.group(2).strip()) if t]
        if len(tokens) != num_gallery:
            errors.append(f"Q{qid}: expected {num_gallery} ids, found {len(tokens)}")
            continue
        try:
            ids = [int(t) for t in tokens]
        except ValueError:
            errors.append(f"Q{qid}: non-integer id in row")
            continue
        if len(set(ids)) != len(ids):
            errors.append(f"Q{qid}: {len(ids) - len(set(ids))} duplicate ids")
        if allowed is not None:
            unknown = sorted(set(ids) - allowed)
            if unknown:
                errors.append(f"Q{qid}: {len(unknown)} ids outside the gallery, e.g. {unknown[:5]}")
        elif min(ids) < 0 or max(ids) >= num_gallery:
            errors.append(f"Q{qid}: id out of range [0, {num_gallery})")
    return errors
