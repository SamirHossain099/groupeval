# groupeval

**Evaluation that respects the group structure benchmarks actually have.**

Most vision and medical-imaging benchmarks are built from *groups* (video frames of one lesion,
slices of one scan, crops of one slide, images of one patient) and then scored as if every file
were an independent observation. Three things go wrong:

| problem | what it looks like | what `groupeval` does |
|---|---|---|
| **duplicates** | the same image in two "independent" test sets, or twice in one | exact pixel audit, no thresholds |
| **splits** | train/test separates files but not groups | group-disjoint splits, leakage asserted |
| **intervals** | confidence intervals computed over files | cluster bootstrap and design effect |
| **resolution** | "not significant", but is that the data or the compute? | variance decomposition and a floor |

```python
from groupeval import audit_duplicates, multi_seed_splits, cluster_ci, naive_ci, paired_group_test

report = audit_duplicates({"train": train_paths, "test_a": a_paths, "test_b": b_paths},
                          to_array=lambda p: np.asarray(Image.open(p)))
print(report.summary())

splits = multi_seed_splits(frame_to_lesion, n_seeds=5, test_fraction=0.2)

honest = cluster_ci(dice_per_frame, lesion_of_frame)
print(honest.width / naive_ci(dice_per_frame).width, "x wider than the file-level interval")

verdict = paired_group_test(dice_model_a, dice_model_b, lesion_of_frame)
print(verdict, "significant" if verdict.excludes_zero else "not distinguishable")
```

## Three things it will tell you that you may not want to hear

**Small correlations are not small problems.** At 25 files per group, an intra-group correlation of
0.1 already inflates the standard error by 1.9×. Whether that matters is arithmetic, not judgement.

**Pairing usually matters more than the correction.** If two methods were scored on the same items,
`paired_group_test` is typically 2–3× sharper than the unpaired form. Using the unpaired test on
paired data can turn a real difference into noise.

**More compute may not help.** On a fixed test set, `decompose_variance` separates the part of your
uncertainty that more training runs shrink from the part they never touch. If the differences your
field reports sit below that floor, the only remedy is more independent groups, and
`required_groups` prices it.

**Unverifiable grouping is not a dead end.** When part of your data ships no grouping metadata,
`independence_sensitivity` sweeps the possibilities and includes the case that needs no assumption
at all: deleting those items, which is strictly more pessimistic than any grouping of them. If your
conclusion survives that row, the unknown stops mattering.

```python
from groupeval import independence_sensitivity
rows = independence_sensitivity(scores, suspect=unlabelled_items, counts=(50, 25, 10))
print(rows[-1])   # the assumption-free worst case
```

**A confident interval is not the same as a robust result.** When one side of a comparison is a
fixed subset of the data, no bootstrap explores which groups landed in it. `leave_one_group_out`
drops each group in turn and tells you whether the sign survives: on the benchmark this package
came from, a contrast of +0.133 with an interval of [+0.128, +0.138] reversed to −0.136 when one
group was removed.

```python
from groupeval import leave_one_group_out
r = leave_one_group_out(scores, groups, mask=in_the_subset)
r["sign_stable"], r["largest_share_fraction"]
```

## Install

```bash
pip install groupeval
```

Depends on numpy only.

## Provenance

Built while auditing a widely used medical-imaging benchmark, where the file-level convention had
kept a duplicated test set invisible for six years and made every published confidence interval two
to four times too narrow. The tools are dataset-agnostic because the mistake is.

MIT licensed.
