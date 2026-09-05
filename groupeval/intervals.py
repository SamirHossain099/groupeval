"""Confidence intervals and comparisons that respect grouping.

A file-level interval on grouped data is too narrow, by the square root of the design effect

    deff = 1 + (m - 1) * rho

for average group size `m` and intra-group correlation `rho`. The important thing about that formula
is how fast it bites: at 25 files per group, a correlation of only 0.1 already inflates the standard
error by 1.9x, and rho = 0.5 inflates it by 3.6x. "The correlation is small" is not a defence.

Two comparison functions, and choosing between them matters more than most users expect:

    `paired_group_test`      both methods scored on the SAME items. The item effect cancels, and
                             the interval is typically 2-3x narrower.
    `two_sample_group_test`  methods scored on different items. Weaker, and correct only when the
                             pairing genuinely does not exist.

Using the unpaired form on paired data is a common and expensive mistake: it can turn a real
difference into "not significant" and an ordering into noise.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n_groups: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def _index_groups(groups: Sequence[Hashable]) -> list[np.ndarray]:
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def _check(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f"{(~np.isfinite(v)).sum()} non-finite values passed to groupeval")
    return v


def icc(values: Sequence[float], groups: Sequence[Hashable]) -> float:
    """One-way intra-class correlation, clipped to [0, 1].

    The ANOVA estimator can go negative when between-group signal is weaker than chance. That is a
    real outcome but not a usable correlation, and a negative value would make the design effect
    less than 1 and *narrow* the interval: turning the correction into a second bug.
    """
    v = _check(values)
    idx = _index_groups(groups)
    k, n = len(idx), len(v)
    if k < 2 or n <= k:
        return 0.0
    grand = v.mean()
    ms_between = sum(len(i) * (v[i].mean() - grand) ** 2 for i in idx) / (k - 1)
    ms_within = sum(((v[i] - v[i].mean()) ** 2).sum() for i in idx) / (n - k)
    sizes = np.array([len(i) for i in idx], dtype=float)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    if ms_within <= 0 or m0 <= 0:
        return 0.0
    between = (ms_between - ms_within) / m0
    return float(np.clip(between / (between + ms_within), 0.0, 1.0))


def design_effect(values: Sequence[float], groups: Sequence[Hashable]) -> dict:
    """deff = 1 + (m - 1) * rho, and the standard-error inflation it implies."""
    v = _check(values)
    idx = _index_groups(groups)
    m = len(v) / len(idx)
    rho = icc(v, groups)
    deff = 1.0 + (m - 1.0) * rho
    return dict(n=len(v), n_groups=len(idx), mean_group_size=m, icc=rho, design_effect=deff,
                se_inflation=float(np.sqrt(deff)),
                worst_case_se_inflation=float(np.sqrt(m)))


def naive_ci(values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
             seed: int = 0) -> Interval:
    """File-level percentile bootstrap, what most benchmarks report. Provided for contrast."""
    v = _check(values)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(v.mean()), float(lo), float(hi), n_groups=len(v))


def cluster_ci(values: Sequence[float], groups: Sequence[Hashable], *,
               n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile bootstrap resampling GROUPS with replacement.

    Larger groups carry more weight, matching how a file-level mean is defined, so this corrects
    the interval without changing the point estimate.
    """
    v = _check(values)
    idx = _index_groups(groups)
    k = len(idx)
    if k < 2:
        raise ValueError(f"cluster_ci needs at least 2 groups, got {k}")
    if k < 5:
        import warnings
        warnings.warn(f"only {k} groups: the cluster bootstrap is itself high-variance here and "
                      "the interval may be unstable in either direction", stacklevel=2)
    rng = np.random.default_rng(seed)
    sums = np.array([v[i].sum() for i in idx])
    sizes = np.array([len(i) for i in idx], dtype=float)
    pick = rng.integers(0, k, size=(n_boot, k))
    means = sums[pick].sum(1) / sizes[pick].sum(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(v.mean()), float(lo), float(hi), n_groups=k)


def paired_group_test(a: Sequence[float], b: Sequence[float], groups: Sequence[Hashable], *,
                      n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Compare two methods scored on the SAME items, resampling groups.

    Operates on the per-item difference, so item difficulty cancels. This is the right test whenever
    both methods saw the same test set, which is almost always, and it is typically 2-3x sharper
    than the unpaired alternative.
    """
    x, y = _check(a), _check(b)
    if x.shape != y.shape:
        raise ValueError(f"paired test needs one score per item from each method; "
                         f"got {x.shape} and {y.shape}")
    return cluster_ci(x - y, groups, n_boot=n_boot, alpha=alpha, seed=seed)


def two_sample_group_test(a: Sequence[float], a_groups: Sequence[Hashable],
                          b: Sequence[float], b_groups: Sequence[Hashable], *,
                          n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Compare two methods scored on DIFFERENT items. Use only when no pairing exists.

    Each side's groups are resampled independently, so item difficulty does not cancel and the
    interval is much wider. If the two methods were in fact evaluated on the same items, use
    `paired_group_test` instead.
    """
    x, y = _check(a), _check(b)
    rng = np.random.default_rng(seed)

    def boot(v, groups):
        idx = _index_groups(groups)
        k = len(idx)
        if k < 2:
            raise ValueError(f"two_sample_group_test needs at least 2 groups per side, got {k}")
        sums = np.array([v[i].sum() for i in idx])
        sizes = np.array([len(i) for i in idx], dtype=float)
        pick = rng.integers(0, k, size=(n_boot, k))
        return sums[pick].sum(1) / sizes[pick].sum(1), k

    ba, ka = boot(x, a_groups)
    bb, kb = boot(y, b_groups)
    diff = ba - bb
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return Interval(float(x.mean() - y.mean()), float(lo), float(hi), n_groups=min(ka, kb))


def leave_one_group_out(values: Sequence[float], groups: Sequence[Hashable],
                        mask: Sequence[bool], *, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    """Does a contrast survive dropping any single group?

    `mask` splits the items in two (say, a duplicated slice against the rest) and the statistic is
    the difference of their means. This recomputes it with each group removed in turn, and reports
    whether the sign holds throughout.

    Use it whenever one side of a comparison is a **fixed** subset of the data rather than something
    you can resample. Resampling items, or even resampling training runs, tells you nothing about
    the "which groups happened to be in the subset" dimension: on a fixed subset that dimension has
    no variability to find, so a bootstrap will report a confident interval around a number that one
    group is carrying.

    This is not a hypothetical failure mode. On the benchmark this package came from, a contrast of
    +0.132 with a bootstrap interval of [+0.122, +0.142] reversed to -0.149 when a single group was
    removed: 82% of the subset was one group, and the interval could not see it.

    Returns the full-sample difference, one entry per dropped group, and `sign_stable`, which is the
    part to check first.
    """
    v = _check(values)
    m = np.asarray(mask, dtype=bool)
    g = np.asarray(groups)
    if not (len(v) == len(m) == len(g)):
        raise ValueError(f"values, groups and mask must be the same length; "
                         f"got {len(v)}, {len(g)}, {len(m)}")
    if m.all() or not m.any():
        raise ValueError("mask must select some items and leave some out")

    def diff(keep):
        a, b = v[keep & m], v[keep & ~m]
        if not len(a) or not len(b):
            return None
        return float(a.mean() - b.mean())

    everything = np.ones(len(v), dtype=bool)
    full = diff(everything)
    out, dropped = {}, []
    for name in sorted(set(g.tolist()), key=str):
        keep = g != name
        d = diff(keep)
        if d is None:
            dropped.append(name)
            continue
        out[name] = d

    if not out:
        raise ValueError("dropping any single group empties one side of the comparison")
    values_ = list(out.values())
    stable = all(np.sign(x) == np.sign(full) for x in values_)
    worst = min(out, key=lambda k: abs(out[k]) if stable else out[k] * np.sign(full))
    # share of the masked side contributed by each group, the usual explanation when it is unstable
    share = {name: float((m & (g == name)).sum() / m.sum()) for name in out}
    return dict(full=full, without=out, sign_stable=bool(stable),
                span=(float(min(values_)), float(max(values_))),
                most_influential=worst, largest_share=max(share, key=share.get),
                largest_share_fraction=float(max(share.values())),
                groups_skipped=dropped, n_groups=len(out))
