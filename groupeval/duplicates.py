"""Exact duplicate detection. No thresholds, nothing to calibrate, nothing to argue with.

Perceptual hashing needs an operating point, and an operating point is a judgement call a reviewer
can dispute. Hashing decoded pixel arrays does not: two images either contain the same pixels or
they do not.

Hashing the *decoded array* rather than the file bytes matters. The same frame re-encoded: a
different PNG compression level, stripped metadata, a round-trip through a redistribution pipeline ,
has different bytes and identical pixels. Byte-level comparison misses exactly the duplicates that
arise from a dataset being repackaged, which are the ones that actually occur.

Three questions, in order of how damaging a positive answer is:

    1. does an image appear in both a training and a test split?
    2. does it appear in two different "independent" test sets?
    3. does it appear twice inside one set?
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable


def pixel_digest(image: Any, to_array: Callable[[Any], Any] | None = None) -> str:
    """Stable digest of an image's pixel content.

    `image` may be a numpy array, or anything `to_array` converts to one. Arrays are made
    C-contiguous first so that a view and its copy digest identically: otherwise two references to
    the same data could hash differently, which would silently under-count duplicates.
    """
    import numpy as np

    arr = to_array(image) if to_array is not None else image
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass
class DuplicateReport:
    """What `audit_duplicates` found. Counts are of distinct *images*, not of files."""

    n_items: int
    n_distinct: int
    within_split: dict[str, dict[str, int]] = field(default_factory=dict)
    cross_split: dict[tuple[str, str], int] = field(default_factory=dict)
    examples: dict[tuple[str, str], list[list[str]]] = field(default_factory=dict)

    @property
    def n_redundant(self) -> int:
        return self.n_items - self.n_distinct

    @property
    def cross_split_duplicates(self) -> int:
        return sum(self.cross_split.values())

    def is_clean(self) -> bool:
        """True when no image appears in more than one split and none is repeated within a split."""
        return self.cross_split_duplicates == 0 and self.n_redundant == 0

    def summary(self) -> str:
        lines = [f"{self.n_items} items -> {self.n_distinct} distinct images "
                 f"({self.n_redundant} redundant)"]
        if self.cross_split:
            lines.append("shared between splits:")
            for (a, b), n in sorted(self.cross_split.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {a} <-> {b}: {n} images")
        else:
            lines.append("no image appears in two splits")
        for split, info in sorted(self.within_split.items()):
            if info["redundant"]:
                lines.append(f"  {split}: {info['n']} files -> {info['distinct']} distinct "
                             f"({info['redundant']} redundant)")
        return "\n".join(lines)


def audit_duplicates(
    splits: Mapping[str, Iterable[Any]],
    *,
    key: Callable[[Any], str] | None = None,
    to_array: Callable[[Any], Any] | None = None,
    name: Callable[[Any], str] | None = None,
    max_examples: int = 5,
) -> DuplicateReport:
    """Audit a mapping of split name -> images for exact duplicates.

    Parameters
    ----------
    splits
        e.g. ``{"train": [...], "test_a": [...], "test_b": [...]}``. Values may be arrays, paths,
        or any object, provided `key` or `to_array` can reduce them to pixel content.
    key
        Optional function returning a digest directly. Use when you already have hashes, or when
        images are too large to hold in memory and you stream them yourself.
    to_array
        Converts an item to a numpy array (e.g. ``lambda p: np.asarray(Image.open(p))``).
    name
        Human-readable label for an item, used in the examples. Defaults to ``str``.

    Returns
    -------
    DuplicateReport

    Notes
    -----
    Memory is one digest per item, not one image: safe on large benchmarks.
    """
    name = name or (lambda x: str(x))
    digest_of = key or (lambda item: pixel_digest(item, to_array))

    by_digest: dict[str, list[tuple[str, str]]] = defaultdict(list)
    per_split_counts: dict[str, int] = {}
    for split, items in splits.items():
        count = 0
        for item in items:
            by_digest[digest_of(item)].append((split, name(item)))
            count += 1
        per_split_counts[split] = count

    within: dict[str, dict[str, int]] = {}
    for split, n in per_split_counts.items():
        distinct = sum(1 for rows in by_digest.values()
                       if any(s == split for s, _ in rows))
        # count files of this split per digest to find repeats inside the split
        repeats = 0
        for rows in by_digest.values():
            k = sum(1 for s, _ in rows if s == split)
            if k > 1:
                repeats += k - 1
        within[split] = dict(n=n, distinct=distinct, redundant=repeats)

    cross: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for rows in by_digest.values():
        present = sorted({s for s, _ in rows})
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair = (present[i], present[j])
                cross[pair] += 1
                if len(examples[pair]) < max_examples:
                    examples[pair].append([f"{s}/{n_}" for s, n_ in rows])

    total = sum(per_split_counts.values())
    return DuplicateReport(n_items=total, n_distinct=len(by_digest),
                           within_split=within, cross_split=dict(cross),
                           examples=dict(examples))


def assert_no_leakage(report: DuplicateReport, train_splits: Sequence[str]) -> None:
    """Raise if any training image also appears in a non-training split.

    Intended for a test suite or a CI step, so contamination fails loudly instead of being
    discovered in review.
    """
    train = set(train_splits)
    offenders = {pair: n for pair, n in report.cross_split.items()
                 if (pair[0] in train) != (pair[1] in train)}
    if offenders:
        detail = ", ".join(f"{a} <-> {b}: {n}" for (a, b), n in offenders.items())
        raise AssertionError(f"train/test image duplication detected, {detail}")
