"""What a benchmark can resolve, and how much data it would take to resolve more.

The question this answers: *is the difference I cannot detect a limit of my compute, or a limit of
the benchmark?* On a fixed test set those are different, and the distinction is usually decisive.

    score[run, group] = mu + a[run] + b[group] + e[run, group]

`a` is training randomness and shrinks as 1/n_runs: more seeds buy it down. `b` is which items are
in the test set, and on a **fixed** benchmark it never shrinks, because retraining does not change
which items those are. So there is a floor:

    floor = sqrt(var_b / n_groups)

If the differences a field reports are below that floor, no amount of compute settles them and the
only remedy is more independent groups. `required_groups` prices that.

A caution learned the hard way: the floor scales as ``n ** -exponent`` with exponent = 0.5 only for **equal-sized
independent groups**. With unequal group sizes the exponent is smaller, measured at 0.44 on one
real benchmark, so `required_groups` accepts the exponent as a parameter and you should measure
it, not assume it. `fit_scaling_exponent` does that by subsampling.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PowerReport:
    n_runs: int
    n_groups: int
    var_run: float
    var_group: float
    var_resid: float

    def se(self, n_runs: int) -> float:
        """Standard error of the overall mean with `n_runs` training runs."""
        return float(np.sqrt(self.var_run / n_runs
                             + self.var_group / self.n_groups
                             + self.var_resid / (n_runs * self.n_groups)))

    @property
    def floor(self) -> float:
        """The standard error as n_runs -> infinity. More compute never goes below this."""
        return float(np.sqrt(self.var_group / self.n_groups))

    @property
    def irreducible_share(self) -> float:
        now = self.se(self.n_runs)
        return float(self.floor / now) if now > 0 else float("nan")

    def summary(self) -> str:
        return "\n".join([
            f"{self.n_runs} runs x {self.n_groups} groups",
            f"  SE now                 {self.se(self.n_runs):.5f}",
            f"  SE floor (inf. runs)   {self.floor:.5f}",
            f"  irreducible share      {self.irreducible_share:.0%}",
        ])


def decompose_variance(scores: Mapping[Hashable, Mapping[Hashable, float]]) -> PowerReport:
    """Two-way random-effects decomposition of ``scores[run][group]``.

    Requires a balanced layout: every run scored on the same groups, which is what a fixed test set
    gives you. Missing cells raise rather than being imputed.
    """
    runs = sorted(scores)
    if len(runs) < 2:
        raise ValueError("need at least 2 runs to separate run variance from group variance")
    group_sets = [set(scores[r]) for r in runs]
    groups = sorted(set.intersection(*group_sets))
    if not groups:
        raise ValueError("runs share no common groups")
    if any(len(gs) != len(groups) for gs in group_sets):
        raise ValueError("unbalanced layout: every run must be scored on the same groups "
                         "(a fixed test set gives this; resampled test sets do not)")

    mat = np.array([[scores[r][g] for g in groups] for r in runs], dtype=float)
    if not np.isfinite(mat).all():
        raise ValueError("non-finite score in the matrix")
    n_r, n_g = mat.shape
    grand = mat.mean()
    ms_run = float(n_g * ((mat.mean(axis=1) - grand) ** 2).sum() / max(n_r - 1, 1))
    ms_group = float(n_r * ((mat.mean(axis=0) - grand) ** 2).sum() / max(n_g - 1, 1))
    resid = mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + grand
    ms_resid = float((resid ** 2).sum() / max((n_r - 1) * (n_g - 1), 1))
    return PowerReport(n_runs=n_r, n_groups=n_g,
                       var_run=max((ms_run - ms_resid) / n_g, 0.0),
                       var_group=max((ms_group - ms_resid) / n_r, 0.0),
                       var_resid=ms_resid)


def resolution_floor(report: PowerReport, *, z: float = 1.96) -> float:
    """Half-width of the interval on a single mean at infinite runs."""
    return z * report.floor


def required_groups(n_now: int, halfwidth_now: float, halfwidth_target: float, *,
                    exponent: float = 0.5) -> float:
    """Groups needed to reach `halfwidth_target`, given today's `halfwidth_now` at `n_now`.

    `exponent` defaults to the textbook 0.5 but **should be measured**: see
    `fit_scaling_exponent`. Unequal group sizes push it below 0.5, which means more groups are
    needed than the default suggests.
    """
    if halfwidth_target <= 0 or halfwidth_now <= 0:
        raise ValueError("half-widths must be positive")
    if not 0.1 < exponent <= 0.5:
        raise ValueError(f"implausible scaling exponent {exponent}; expected 0.1 < p <= 0.5")
    return float(n_now * (halfwidth_now / halfwidth_target) ** (1.0 / exponent))


def fit_scaling_exponent(halfwidth_at: Callable[[int], float],
                         sizes: Sequence[int]) -> dict:
    """Fit ``hw ~ C * n**-p`` from measured half-widths at several group counts.

    `halfwidth_at(n)` should subsample `n` groups and return the interval half-width. Fitting on
    real subsamples is the only way to know whether the 1/sqrt(n) assumption holds for a particular
    dataset's group-size distribution.
    """
    sizes = [int(n) for n in sizes]
    if len(sizes) < 3:
        raise ValueError("need at least 3 sizes to fit an exponent")
    hws = np.array([halfwidth_at(n) for n in sizes], dtype=float)
    if not np.isfinite(hws).all() or (hws <= 0).any():
        raise ValueError("halfwidth_at returned a non-positive or non-finite value")
    slope, intercept = np.polyfit(np.log(sizes), np.log(hws), 1)
    p = float(-slope)
    predicted = hws[-1] * (sizes[-1] / np.array(sizes, dtype=float)) ** 0.5
    return dict(exponent=p, intercept=float(np.exp(intercept)),
                sizes=list(sizes), halfwidths=[float(h) for h in hws],
                max_deviation_from_sqrt=float(np.max(np.abs(hws - predicted) / predicted)),
                close_to_sqrt=bool(abs(p - 0.5) < 0.08))


def merge_groups(scores: Mapping[Hashable, Mapping[Hashable, float]],
                 mapping: Mapping[Hashable, Hashable],
                 weights: Mapping[Hashable, float] | None = None) -> dict:
    """Combine groups that `mapping` sends to the same key, averaging their scores.

    Use this when groups you treated as independent may actually belong together: several video
    clips of one lesion, several slides from one patient. Groups absent from `mapping` are kept
    as they are.

    `weights` gives each original group's size, so merged groups are a weighted mean rather than a
    mean of means. Pass it whenever the underlying groups differ in size; leaving it out silently
    treats a 2-frame group and a 200-frame group as equals, which is a real error and not a small
    one.
    """
    out = {}
    for run, per_group in scores.items():
        acc: dict = {}
        for g, v in per_group.items():
            key = mapping.get(g, g)
            w = float(weights.get(g, 1.0)) if weights else 1.0
            if w < 0:
                raise ValueError(f"negative weight for group {g!r}")
            num, den = acc.get(key, (0.0, 0.0))
            acc[key] = (num + w * float(v), den + w)
        out[run] = {k: n / d for k, (n, d) in acc.items() if d > 0}
    return out


def floor_excluding(scores: Mapping[Hashable, Mapping[Hashable, float]],
                    exclude: Sequence[Hashable]) -> PowerReport:
    """The resolution floor with a set of groups deleted.

    This is the assumption-free bound for groups whose independence you cannot verify. Deleting a
    group is strictly more pessimistic than merging it with another, because even one merged group
    still contributes one independent unit, so if your conclusion survives here, it survives any
    grouping of those items and you do not have to defend a guess about how many there really are.
    """
    drop = set(exclude)
    filtered = {r: {g: v for g, v in per.items() if g not in drop} for r, per in scores.items()}
    if not any(filtered.values()):
        raise ValueError("excluding those groups leaves nothing to decompose")
    return decompose_variance(filtered)


def independence_sensitivity(scores: Mapping[Hashable, Mapping[Hashable, float]],
                             suspect: Sequence[Hashable],
                             *, counts: Sequence[int] = (), n_rep: int = 25, seed: int = 0,
                             weights: Mapping[Hashable, float] | None = None,
                             z: float = 1.96) -> list:
    """Sweep the floor against the possibility that `suspect` groups are not independent.

    Returns one row per scenario, from "as reported" down to "these groups contribute nothing",
    each with the resolution floor and the half-width it implies. Read it as a robustness check:
    if the effect you care about is below the floor in the *last* row, no assumption about the
    suspect groups can rescue it, and the question of how many there really are stops mattering.

    `counts` are the hypothetical true group counts to test. The assignment at each count is
    **random**, repeated `n_rep` times and reported at the median. Random assignment understates
    the damage, items that genuinely share a group resemble each other more than randomly paired
    ones do, so the intermediate rows are optimistic and the honest weight of the argument sits on
    the final, assignment-free row.
    """
    rng = np.random.default_rng(seed)
    suspect = list(suspect)
    if not suspect:
        raise ValueError("no suspect groups given")
    missing = set(suspect) - set().union(*(set(p) for p in scores.values()))
    if missing:
        raise ValueError(f"{len(missing)} suspect groups are not in `scores`, e.g. {list(missing)[:3]}")

    rows = [dict(scenario="as reported", n_suspect_groups=len(suspect),
                 floor=decompose_variance(scores).floor)]
    for k in counts:
        k = int(k)
        if k < 1:
            raise ValueError(f"group count must be at least 1, got {k}")
        if k >= len(suspect):
            continue
        floors = []
        for _ in range(n_rep):
            order = list(suspect)
            rng.shuffle(order)
            m = {g: f"__merged{i}" for i, g in enumerate(order[:k])}
            for g in order[k:]:
                m[g] = f"__merged{int(rng.integers(k))}"
            floors.append(decompose_variance(merge_groups(scores, m, weights)).floor)
        rows.append(dict(scenario=f"{k} groups (random assignment)", n_suspect_groups=k,
                         floor=float(np.median(floors))))
    rows.append(dict(scenario="excluded entirely (worst case)", n_suspect_groups=0,
                     floor=floor_excluding(scores, suspect).floor))
    for r in rows:
        r["halfwidth"] = z * r["floor"]
    return rows
