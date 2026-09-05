"""Group-disjoint train/test splits, with the leakage assertion built in.

A split that separates *files* but not *groups* puts the same lesion, scan or patient on both sides.
The fix is not complicated, hold out whole groups, but two details are easy to get wrong and both
matter:

    **Multi-seed is not optional.** When a dataset has 15 groups, a single group-disjoint split has
    far more variance than a file-level one, because it depends on which handful of groups landed in
    the test set. A one-off comparison confounds the leakage correction with fold luck. These
    functions default to producing several seeds and the API makes the single-split case awkward on
    purpose.

    **The size target cannot be hit exactly.** Whole groups do not divide evenly into a frame
    target. The achieved size is reported rather than silently accepted, so a difference in score
    can never be quietly attributed to a difference in test-set size.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass
class Split:
    """One group-disjoint partition."""

    seed: int
    train: list[Hashable]
    test: list[Hashable]
    test_groups: list[Hashable]
    n_groups: int

    @property
    def leakage(self) -> float:
        """Always 0.0, kept as an attribute so callers can assert on it explicitly."""
        return 0.0

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)


def leakage_fraction(train: Iterable[Hashable], test: Iterable[Hashable],
                     groups: Mapping[Hashable, Hashable]) -> float:
    """Fraction of test items whose group also appears in train.

    This is the number that is 0.0 for a group-disjoint split and often 1.0 for a file-level one.
    Use it to audit a split you did not build.
    """
    train_groups = {groups[i] for i in train if i in groups}
    test_items = [i for i in test]
    if not test_items:
        return 0.0
    hits = sum(1 for i in test_items if groups.get(i) in train_groups)
    return hits / len(test_items)


def group_disjoint_split(groups: Mapping[Hashable, Hashable], *,
                         test_fraction: float = 0.2,
                         seed: int = 0,
                         tolerance: float = 0.35) -> Split:
    """Hold out whole groups until roughly `test_fraction` of items are in the test side.

    Greedy over a shuffled group order rather than optimised to hit the target exactly: an
    optimiser would return nearly the same split for every seed, which defeats the purpose of
    running several.

    Raises
    ------
    ValueError
        If the data has fewer than two groups, or the achieved test size misses the target by more
        than `tolerance`. Failing loudly beats returning a lopsided split that looks fine.
    """
    by_group: dict[Hashable, list[Hashable]] = defaultdict(list)
    for item, g in groups.items():
        by_group[g].append(item)
    if len(by_group) < 2:
        raise ValueError(f"need at least 2 groups to build a group-disjoint split, "
                         f"got {len(by_group)}")

    n_items = len(groups)
    target = n_items * test_fraction
    order = sorted(by_group)
    random.Random(seed).shuffle(order)

    test_groups: list[Hashable] = []
    n = 0
    for g in order:
        if n >= target:
            break
        test_groups.append(g)
        n += len(by_group[g])

    chosen = set(test_groups)
    test = [i for i, g in groups.items() if g in chosen]
    train = [i for i, g in groups.items() if g not in chosen]
    if not train or not test:
        raise ValueError("split left one side empty; try a different test_fraction")
    achieved = len(test) / n_items
    if abs(achieved - test_fraction) > tolerance * test_fraction:
        raise ValueError(
            f"achieved test fraction {achieved:.3f} misses target {test_fraction:.3f} by more than "
            f"{tolerance:.0%}; whole groups cannot hit it. Widen `tolerance` and report the "
            "achieved size, or use a dataset with smaller groups.")

    split = Split(seed=seed, train=train, test=test, test_groups=test_groups,
                  n_groups=len(by_group))
    assert leakage_fraction(train, test, groups) == 0.0, "group-disjoint split leaked"
    return split


def multi_seed_splits(groups: Mapping[Hashable, Hashable], *,
                      n_seeds: int = 5,
                      test_fraction: float = 0.2,
                      tolerance: float = 0.35) -> list[Split]:
    """Several group-disjoint splits. This is the intended entry point.

    With few groups the between-split variance dominates, so a single split is not a measurement.
    Reporting the spread across seeds is the point, not a nicety.
    """
    if n_seeds < 2:
        raise ValueError("multi_seed_splits needs n_seeds >= 2; a single group-disjoint split "
                         "confounds the correction with fold luck")
    return [group_disjoint_split(groups, test_fraction=test_fraction, seed=s,
                                 tolerance=tolerance) for s in range(n_seeds)]


def summarise(splits: Sequence[Split]) -> str:
    lines = [f"{len(splits)} group-disjoint splits over {splits[0].n_groups} groups"]
    for s in splits:
        tr, te = s.sizes
        lines.append(f"  seed {s.seed}: {tr} train / {te} test  "
                     f"({len(s.test_groups)} held-out groups, leakage {s.leakage:.0%})")
    sizes = [s.sizes[1] for s in splits]
    lines.append(f"  test size varies {min(sizes)}-{max(sizes)}, whole groups cannot hit a "
                 "frame target exactly; report it rather than tuning it away")
    return "\n".join(lines)
