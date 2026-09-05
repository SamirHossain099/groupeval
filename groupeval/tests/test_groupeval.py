"""Tests for the groupeval package.

The package exists to stop people reporting over-confident numbers, so its own failure modes are
the dangerous ones: a duplicate audit that misses duplicates, a "group-disjoint" split that leaks,
or an interval that is merely wider rather than correct. Each is tested against constructed data
with a known answer, and the null cases are tested as hard as the positive ones: a tool that always
says "duplicated" or always says "not significant" would pass a one-sided test suite.
"""
import numpy as np
import pytest

from groupeval import (
    audit_duplicates,
    cluster_ci,
    decompose_variance,
    design_effect,
    floor_excluding,
    group_disjoint_split,
    independence_sensitivity,
    leakage_fraction,
    leave_one_group_out,
    merge_groups,
    multi_seed_splits,
    naive_ci,
    paired_group_test,
    pixel_digest,
    required_groups,
    resolution_floor,
    two_sample_group_test,
)
from groupeval.duplicates import assert_no_leakage
from groupeval.power import fit_scaling_exponent


def img(seed, h=8, w=8):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- duplicates
def test_identical_pixels_digest_identically_regardless_of_memory_layout():
    a = img(0)
    assert pixel_digest(a) == pixel_digest(a.copy())
    assert pixel_digest(a) == pixel_digest(a[::1])          # a view of the whole array
    assert pixel_digest(a) != pixel_digest(img(1))


def test_finds_an_image_shared_between_two_test_sets():
    shared = img(10)
    r = audit_duplicates({"test_a": [shared, img(11)], "test_b": [shared, img(12)]})
    assert r.cross_split == {("test_a", "test_b"): 1}
    assert r.cross_split_duplicates == 1
    assert not r.is_clean()


def test_reports_clean_when_there_are_no_duplicates():
    """The null case. A detector that always fires is worthless."""
    r = audit_duplicates({"train": [img(i) for i in range(5)],
                          "test": [img(i) for i in range(100, 105)]})
    assert r.cross_split == {}
    assert r.is_clean()
    assert r.n_distinct == r.n_items == 10


def test_counts_repeats_within_a_single_split():
    dup = img(20)
    r = audit_duplicates({"test": [dup, dup, dup, img(21)]})
    assert r.within_split["test"] == dict(n=4, distinct=2, redundant=2)
    assert r.n_redundant == 2


def test_assert_no_leakage_raises_only_across_the_train_boundary():
    shared = img(30)
    r = audit_duplicates({"train": [shared], "test": [shared]})
    with pytest.raises(AssertionError, match="train/test image duplication"):
        assert_no_leakage(r, ["train"])
    # two test sets sharing an image is a different problem and must not trip this check
    r2 = audit_duplicates({"train": [img(31)], "test_a": [shared], "test_b": [shared]})
    assert_no_leakage(r2, ["train"])


def test_audit_accepts_a_precomputed_key():
    r = audit_duplicates({"a": ["x", "y"], "b": ["y", "z"]}, key=lambda s: s)
    assert r.cross_split == {("a", "b"): 1}


# --------------------------------------------------------------------------- splits
def groups_of(n_groups, per_group):
    return {f"item{g}_{i}": f"grp{g}" for g in range(n_groups) for i in range(per_group)}


def test_group_disjoint_split_never_leaks():
    g = groups_of(20, 10)
    for seed in range(8):
        s = group_disjoint_split(g, test_fraction=0.25, seed=seed)
        assert leakage_fraction(s.train, s.test, g) == 0.0
        assert not set(s.train) & set(s.test)


def test_a_file_level_split_does_leak_and_the_metric_says_so():
    """Guards the metric itself: if leakage_fraction always returned 0, the tests above pass.

    The split must be RANDOM over items. Taking a sorted prefix instead would land almost on group
    boundaries, `item0_*` sorts before `item1_*`, and produce a nearly group-disjoint split by
    accident, which is exactly what this test needs not to do.
    """
    g = groups_of(10, 20)
    items = sorted(g)
    import random as _random
    _random.Random(0).shuffle(items)
    train, test = items[:150], items[150:]
    assert leakage_fraction(train, test, g) > 0.9


def test_seeds_produce_different_splits():
    g = groups_of(20, 10)
    held = [tuple(sorted(group_disjoint_split(g, seed=s).test_groups)) for s in range(6)]
    assert len(set(held)) >= 5, held


def test_split_refuses_when_there_are_too_few_groups():
    with pytest.raises(ValueError, match="at least 2 groups"):
        group_disjoint_split({"a": "g", "b": "g"})


def test_split_refuses_a_target_whole_groups_cannot_hit():
    """One enormous group means no split is close to the target, that must fail loudly."""
    g = {f"i{i}": ("big" if i < 95 else f"small{i}") for i in range(100)}
    with pytest.raises(ValueError, match="misses target"):
        group_disjoint_split(g, test_fraction=0.5, tolerance=0.1)


def test_multi_seed_requires_more_than_one_seed():
    with pytest.raises(ValueError, match="n_seeds >= 2"):
        multi_seed_splits(groups_of(10, 5), n_seeds=1)


def test_multi_seed_returns_the_requested_number():
    assert len(multi_seed_splits(groups_of(15, 8), n_seeds=4)) == 4


# --------------------------------------------------------------------------- intervals
def clustered(rng, n_groups=15, size=25, mu=0.7, between=0.12, within=0.03):
    offs = rng.normal(0, between, n_groups)
    vals = np.concatenate([mu + offs[c] + rng.normal(0, within, size) for c in range(n_groups)])
    grp = np.repeat(np.arange(n_groups), size)
    return vals, grp


def test_design_effect_matches_the_closed_form():
    v, g = clustered(np.random.default_rng(0))
    d = design_effect(v, g)
    assert d["design_effect"] == pytest.approx(1 + (d["mean_group_size"] - 1) * d["icc"])
    assert d["se_inflation"] <= d["worst_case_se_inflation"] + 1e-9


def test_cluster_interval_is_wider_on_clustered_data():
    v, g = clustered(np.random.default_rng(1))
    assert cluster_ci(v, g, n_boot=2000).width > 2 * naive_ci(v, n_boot=2000).width


def test_cluster_interval_matches_the_naive_one_when_every_group_is_a_singleton():
    """The correction must not add width where there is no clustering."""
    v = np.random.default_rng(2).normal(0.7, 0.1, 200)
    g = np.arange(200)
    assert cluster_ci(v, g, n_boot=4000).width == pytest.approx(
        naive_ci(v, n_boot=4000).width, rel=0.1)


def test_cluster_ci_warns_with_very_few_groups():
    v, g = clustered(np.random.default_rng(3), n_groups=3, size=10)
    with pytest.warns(UserWarning, match="high-variance"):
        cluster_ci(v, g, n_boot=500)


def test_paired_test_detects_a_uniform_improvement():
    v, g = clustered(np.random.default_rng(4))
    assert paired_group_test(v + 0.05, v, g, n_boot=2000).excludes_zero


def test_paired_test_ignores_a_difference_confined_to_one_group():
    """25 winning files look decisive; one group out of 15 is not."""
    v, g = clustered(np.random.default_rng(5))
    a = v.copy()
    a[g == 3] += 0.6
    assert not paired_group_test(a, v, g, n_boot=4000).excludes_zero


def test_paired_is_sharper_than_unpaired_on_the_same_data():
    """The reason the distinction is documented so loudly."""
    v, g = clustered(np.random.default_rng(6))
    a = v + 0.03
    paired = paired_group_test(a, v, g, n_boot=3000)
    unpaired = two_sample_group_test(a, g, v, g, n_boot=3000)
    assert paired.width < unpaired.width / 2


def test_interval_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        cluster_ci([1.0, np.nan], [0, 1])


# --------------------------------------------------------------------------- power
def synth_scores(n_runs, n_groups, sd_run, sd_group, sd_resid, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, sd_run, n_runs)
    b = rng.normal(0, sd_group, n_groups)
    e = rng.normal(0, sd_resid, (n_runs, n_groups))
    return {r: {g: 0.8 + a[r] + b[g] + e[r, g] for g in range(n_groups)} for r in range(n_runs)}


def test_decomposition_recovers_known_components():
    ests = [decompose_variance(synth_scores(40, 60, 0.05, 0.10, 0.02, seed=k)) for k in range(20)]
    assert np.mean([np.sqrt(r.var_run) for r in ests]) == pytest.approx(0.05, abs=0.015)
    assert np.mean([np.sqrt(r.var_group) for r in ests]) == pytest.approx(0.10, abs=0.015)


def test_floor_is_near_zero_when_only_the_run_varies():
    r = decompose_variance(synth_scores(30, 50, 0.08, 0.0, 0.005, seed=1))
    assert r.irreducible_share < 0.15


def test_floor_dominates_when_only_the_groups_vary():
    r = decompose_variance(synth_scores(30, 50, 0.0, 0.10, 0.005, seed=2))
    assert r.irreducible_share > 0.95


def test_more_runs_never_widen_the_standard_error():
    r = decompose_variance(synth_scores(20, 40, 0.05, 0.05, 0.02, seed=3))
    ses = [r.se(n) for n in (1, 5, 20, 1000)]
    assert all(ses[i] >= ses[i + 1] for i in range(len(ses) - 1))
    assert ses[-1] >= r.floor


def test_decomposition_refuses_an_unbalanced_layout():
    s = synth_scores(4, 10, 0.05, 0.05, 0.01)
    del s[0][3]
    with pytest.raises(ValueError, match="unbalanced"):
        decompose_variance(s)


def test_required_groups_scales_as_the_inverse_square_by_default():
    assert required_groups(100, 0.02, 0.01) == pytest.approx(400)
    assert required_groups(100, 0.01, 0.01) == pytest.approx(100)


def test_required_groups_needs_more_when_the_exponent_is_smaller():
    """Unequal group sizes flatten the exponent, and that means MORE groups, not fewer."""
    assert required_groups(175, 0.0074, 0.0058, exponent=0.44) > \
        required_groups(175, 0.0074, 0.0058, exponent=0.5)


def test_required_groups_refuses_an_implausible_exponent():
    with pytest.raises(ValueError, match="implausible scaling exponent"):
        required_groups(100, 0.02, 0.01, exponent=0.9)


def test_fit_scaling_exponent_recovers_one_half_on_ideal_data():
    fit = fit_scaling_exponent(lambda n: 0.5 / np.sqrt(n), [20, 50, 100, 200])
    assert fit["exponent"] == pytest.approx(0.5, abs=0.01)
    assert fit["close_to_sqrt"]


def test_resolution_floor_is_the_scaled_floor():
    r = decompose_variance(synth_scores(10, 30, 0.02, 0.05, 0.01, seed=4))
    assert resolution_floor(r) == pytest.approx(1.96 * r.floor)


# --------------------------------------------------------------------------- end to end
def test_package_reproduces_the_finding_it_was_extracted_from():
    """The generic API, with no domain-specific code, must find what the audit found.

    This is the strongest check available on the package: it re-derives a real result, that
    CVC-300 and CVC-ColonDB share 58 distinct images, and CVC-ColonDB repeats 34 of its own frames ,
    from raw pixels, through the public interface only. Skips when the archive is absent.
    """
    import io
    import os
    import zipfile

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    archive = os.path.join(here, "data", "TestDataset.zip")
    if not os.path.exists(archive):
        pytest.skip("benchmark archive not present")
    Image = pytest.importorskip("PIL.Image")

    with zipfile.ZipFile(archive) as z:
        def load(sub):
            return [np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
                    for n in sorted(z.namelist())
                    if f"{sub}/images" in n and n.lower().endswith(".png")]

        report = audit_duplicates({"CVC-300": load("CVC-300"),
                                   "CVC-ColonDB": load("CVC-ColonDB")})

    assert report.cross_split[("CVC-300", "CVC-ColonDB")] == 58
    assert report.within_split["CVC-ColonDB"]["redundant"] == 34
    assert report.within_split["CVC-300"]["redundant"] == 2
    assert not report.is_clean()


# --------------------------------------------------------------- independence sensitivity
def _panel(n_runs, groups, var_run, var_group, var_resid, rng):
    a = rng.normal(0, np.sqrt(var_run), n_runs)
    b = dict(zip(groups, rng.normal(0, np.sqrt(var_group), len(groups))))
    return {r: {g: 0.8 + a[r] + b[g] + rng.normal(0, np.sqrt(var_resid))
                for g in groups} for r in range(n_runs)}


def test_merge_groups_averages_and_keeps_untouched_groups():
    scores = {0: {"a": 1.0, "b": 3.0, "c": 10.0}}
    out = merge_groups(scores, {"a": "ab", "b": "ab"})
    assert out[0] == pytest.approx({"ab": 2.0, "c": 10.0})


def test_merge_groups_weights_by_group_size():
    """A 2-item group and a 200-item group are not equals; ignoring that is a real error."""
    scores = {0: {"a": 1.0, "b": 3.0}}
    plain = merge_groups(scores, {"a": "m", "b": "m"})
    weighted = merge_groups(scores, {"a": "m", "b": "m"}, weights={"a": 1.0, "b": 9.0})
    assert plain[0]["m"] == pytest.approx(2.0)
    assert weighted[0]["m"] == pytest.approx(2.8)


def test_merge_groups_rejects_a_negative_weight():
    with pytest.raises(ValueError, match="negative weight"):
        merge_groups({0: {"a": 1.0}}, {"a": "m"}, weights={"a": -1.0})


def test_excluding_groups_raises_the_floor():
    """Fewer independent groups must mean a higher floor. The opposite would be a sign error."""
    rng = np.random.default_rng(0)
    groups = [f"g{i}" for i in range(40)]
    scores = _panel(8, groups, 1e-6, 4e-4, 1e-4, rng)
    base = decompose_variance(scores).floor
    kept = floor_excluding(scores, groups[:20]).floor
    assert kept > base * 1.2


def test_excluding_everything_is_refused_rather_than_returning_nan():
    scores = {0: {"a": 0.5}, 1: {"a": 0.6}}
    with pytest.raises(ValueError, match="leaves nothing"):
        floor_excluding(scores, ["a"])


def test_independence_sensitivity_is_monotone_and_ends_at_the_worst_case():
    """The sweep's entire argument is that the last row is the bound. Pin the ordering."""
    rng = np.random.default_rng(1)
    solid = [f"s{i}" for i in range(30)]
    suspect = [f"q{i}" for i in range(40)]
    scores = _panel(8, solid + suspect, 1e-6, 4e-4, 1e-4, rng)
    rows = independence_sensitivity(scores, suspect, counts=(20, 10, 4), n_rep=15, seed=0)

    assert [r["scenario"] for r in rows][0] == "as reported"
    assert rows[-1]["scenario"].startswith("excluded entirely")
    floors = [r["floor"] for r in rows]
    assert floors == sorted(floors), floors
    assert rows[-1]["floor"] == pytest.approx(floor_excluding(scores, suspect).floor)
    for r in rows:
        assert r["halfwidth"] == pytest.approx(1.96 * r["floor"])


def test_independence_sensitivity_skips_counts_that_are_not_a_reduction():
    rows = independence_sensitivity(
        _panel(6, [f"g{i}" for i in range(20)], 1e-6, 4e-4, 1e-4, np.random.default_rng(2)),
        [f"g{i}" for i in range(10)], counts=(50, 5), n_rep=5)
    assert [r["n_suspect_groups"] for r in rows] == [10, 5, 0], "a count of 50 is not a merge"


def test_independence_sensitivity_rejects_groups_that_do_not_exist():
    """Silently ignoring a typo'd group name would report the base case as the worst case."""
    scores = _panel(4, ["a", "b", "c"], 1e-6, 4e-4, 1e-4, np.random.default_rng(3))
    with pytest.raises(ValueError, match="not in `scores`"):
        independence_sensitivity(scores, ["a", "typo"], counts=(1,))
    with pytest.raises(ValueError, match="no suspect groups"):
        independence_sensitivity(scores, [])


def test_independence_sensitivity_rejects_a_zero_group_count():
    scores = _panel(4, [f"g{i}" for i in range(8)], 1e-6, 4e-4, 1e-4, np.random.default_rng(4))
    with pytest.raises(ValueError, match="at least 1"):
        independence_sensitivity(scores, [f"g{i}" for i in range(4)], counts=(0,))


# ------------------------------------------------------------------ leave_one_group_out
def test_leave_one_group_out_detects_a_contrast_carried_by_one_group():
    """The F36 pattern: 82% of the masked side is one group, and dropping it flips the sign."""
    # group "big" is entirely inside the mask and scores high; the rest is unremarkable
    values = [0.95] * 40 + [0.70] * 5 + [0.72] * 55
    groups = ["big"] * 40 + ["small"] * 5 + ["rest"] * 55
    mask = [True] * 45 + [False] * 55
    r = leave_one_group_out(values, groups, mask)
    assert r["full"] > 0
    assert not r["sign_stable"]
    assert r["without"]["big"] < 0, r["without"]
    assert r["largest_share"] == "big"
    assert r["largest_share_fraction"] == pytest.approx(40 / 45)


def test_leave_one_group_out_passes_a_contrast_spread_across_groups():
    rng = np.random.default_rng(0)
    values, groups, mask = [], [], []
    for gi in range(12):
        for j in range(10):
            inside = j < 5
            values.append(0.80 + (0.10 if inside else 0.0) + rng.normal(0, 0.01))
            groups.append(f"g{gi}")
            mask.append(inside)
    r = leave_one_group_out(values, groups, mask)
    assert r["sign_stable"], r["without"]
    assert r["full"] == pytest.approx(0.10, abs=0.02)
    lo, hi = r["span"]
    assert hi - lo < 0.02, "no single group should move a well-spread contrast much"


def test_leave_one_group_out_rejects_a_degenerate_mask():
    with pytest.raises(ValueError, match="some items"):
        leave_one_group_out([1.0, 2.0], ["a", "b"], [True, True])
    with pytest.raises(ValueError, match="some items"):
        leave_one_group_out([1.0, 2.0], ["a", "b"], [False, False])


def test_leave_one_group_out_checks_lengths():
    with pytest.raises(ValueError, match="same length"):
        leave_one_group_out([1.0, 2.0, 3.0], ["a", "b"], [True, False, True])


def test_leave_one_group_out_skips_groups_whose_removal_empties_a_side():
    """With only one group on the masked side, dropping it leaves nothing to compare; that group
    is reported as skipped rather than silently producing a nonsense number."""
    values = [0.9, 0.9, 0.5, 0.5]
    groups = ["only", "only", "other", "other"]
    mask = [True, True, False, False]
    with pytest.raises(ValueError, match="empties one side"):
        leave_one_group_out(values, groups, mask)
