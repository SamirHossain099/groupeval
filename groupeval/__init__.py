"""groupeval: evaluation that respects the group structure benchmarks actually have.

Most vision and medical-imaging benchmarks are built from *groups*: video frames from one lesion,
slices from one scan, crops from one slide, images of one patient. Scores are then reported as if
every file were an independent observation. Three things go wrong, and this package measures and
corrects all three:

    duplicates   the same image appearing twice: across splits, across "independent" test sets,
                 or inside one set. Exact pixel comparison, no thresholds.
    splits       train/test partitions that separate files but not groups, so the test group was
                 seen in training.
    intervals    confidence intervals computed over files when the independent units are groups.

Everything here is dataset-agnostic: you supply the grouping, it does the rest.

Quick start
-----------
>>> from groupeval import audit_duplicates, group_disjoint_split, cluster_ci, paired_group_test
>>> report = audit_duplicates({"train": train_images, "test": test_images})
>>> report.cross_split_duplicates
0
>>> split = group_disjoint_split(groups, test_fraction=0.2, seed=0)
>>> split.leakage        # 0.0 by construction, and asserted
0.0
>>> ci = cluster_ci(scores, groups)
>>> ci.width / naive_ci(scores).width      # how much too narrow the file-level interval was
2.7

Determinism
-----------
Every function returns the same numbers in any process. Bootstraps take an explicit `seed`, and
nothing derives one from Python's builtin `hash()`, which is randomised per interpreter. This is
checked across processes in `groupeval/tests`, and it is stated because the project this package
came from had exactly that bug: two runs of one command gave different confidence intervals, and
nothing failed or warned.

Why it exists
-------------
Built while auditing a widely used medical-imaging benchmark, where the file-level convention had
made a duplicated test set invisible for six years and made every reported confidence interval
two to four times too narrow. The tools are general because the mistake is.

MIT licensed. See CITATION.cff for how to cite.
"""
from groupeval.duplicates import (
    DuplicateReport,
    audit_duplicates,
    pixel_digest,
)
from groupeval.intervals import (
    Interval,
    cluster_ci,
    design_effect,
    leave_one_group_out,
    naive_ci,
    paired_group_test,
    two_sample_group_test,
)
from groupeval.power import (
    PowerReport,
    decompose_variance,
    fit_scaling_exponent,
    floor_excluding,
    independence_sensitivity,
    merge_groups,
    required_groups,
    resolution_floor,
)
from groupeval.splits import (
    Split,
    group_disjoint_split,
    leakage_fraction,
    multi_seed_splits,
)

__version__ = "0.1.0"

__all__ = [
    "DuplicateReport",
    "Interval",
    "PowerReport",
    "Split",
    "audit_duplicates",
    "cluster_ci",
    "decompose_variance",
    "design_effect",
    "fit_scaling_exponent",
    "floor_excluding",
    "group_disjoint_split",
    "independence_sensitivity",
    "leakage_fraction",
    "leave_one_group_out",
    "merge_groups",
    "multi_seed_splits",
    "naive_ci",
    "paired_group_test",
    "pixel_digest",
    "required_groups",
    "resolution_floor",
    "two_sample_group_test",
]
