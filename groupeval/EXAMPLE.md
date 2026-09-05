# A worked example

The shortest path from "I have a grouped benchmark" to "I know what it can resolve".

## 1. Are there duplicates?

```python
import numpy as np
from PIL import Image
from groupeval import audit_duplicates

report = audit_duplicates(
    {"train": train_paths, "test_a": a_paths, "test_b": b_paths},
    to_array=lambda p: np.asarray(Image.open(p).convert("RGB")),
    name=lambda p: p.name,
)
print(report.summary())
```

```
798 items -> 738 distinct images (60 redundant)
shared between splits:
  test_a <-> test_b: 58 images
  test_b: 380 files -> 346 distinct (34 redundant)
```

Two "independent" test sets sharing 58 images means any average over them double-counts. Put
`assert_no_leakage(report, ["train"])` in your test suite so contamination fails loudly.

## 2. Build a split that holds out groups, not files

```python
from groupeval import multi_seed_splits, leakage_fraction
from groupeval.splits import summarise

splits = multi_seed_splits(frame_to_lesion, n_seeds=5, test_fraction=0.2)
print(summarise(splits))
print("existing split leaks:", leakage_fraction(old_train, old_test, frame_to_lesion))
```

`multi_seed_splits` refuses a single seed on purpose: with few groups, one split confounds the
leakage correction with which groups happened to be held out.

## 3. Put an honest interval on the score

```python
from groupeval import cluster_ci, naive_ci, design_effect

d = design_effect(dice, lesion_of_frame)
print(f"rho={d['icc']:.2f}  mean group={d['mean_group_size']:.0f}  "
      f"SE inflation={d['se_inflation']:.1f}x")

print("file-level ", naive_ci(dice))
print("group-level", cluster_ci(dice, lesion_of_frame))
```

If those two intervals differ a lot, every published number on your benchmark is over-confident by
that factor.

## 4. Compare two methods: paired, if they saw the same items

```python
from groupeval import paired_group_test

v = paired_group_test(dice_a, dice_b, lesion_of_frame)
print(v, "distinguishable" if v.excludes_zero else "NOT distinguishable")
```

Use `two_sample_group_test` only when no pairing exists. Pairing is typically 2–3× sharper, and
using the unpaired form on paired data can turn a real difference into noise.

## 5. Is "not significant" a data problem or a compute problem?

```python
from groupeval import decompose_variance, resolution_floor, required_groups

power = decompose_variance({seed: {lesion: mean_dice} for seed in seeds})
print(power.summary())
print("floor:", resolution_floor(power))

print("groups needed for +/-0.006:",
      required_groups(power.n_groups, resolution_floor(power), 0.006, exponent=0.44))
```

On a **fixed** test set, `var_group` never shrinks however many models you train. If the differences
your field reports sit below `resolution_floor`, more compute will not settle them, and
`required_groups` tells you how much more data would.

**Measure the exponent, don't assume it.** `fit_scaling_exponent` subsamples your own groups and
fits it; unequal group sizes push it below 0.5, and on the benchmark this package came from the true
value was 0.44, making the default a 7% under-estimate.

## 6. What if some of your groups aren't really independent?

Almost every benchmark has a set of items with no grouping metadata at all: files that *might* be
one lesion, one patient, one scan, and you cannot tell. The usual response is to count them as
independent and add a sentence to the limitations. You can do better than a sentence:

```python
from groupeval import independence_sensitivity

for row in independence_sensitivity(scores, suspect=unlabelled_items,
                                    counts=(50, 25, 10), n_rep=25):
    print(f"{row['scenario']:34s} floor {row['floor']:.5f}  +/-{row['halfwidth']:.5f}")
```

```
as reported                        floor 0.00408  +/-0.00799
50 groups (random assignment)      floor 0.00477  +/-0.00936
25 groups (random assignment)      floor 0.00536  +/-0.01051
10 groups (random assignment)      floor 0.00608  +/-0.01192
excluded entirely (worst case)     floor 0.00683  +/-0.01339
```

**The last row is the one that matters.** Deleting the suspect items is strictly more pessimistic
than any grouping of them, even a single merged group still contributes one independent unit, so
if the effect you care about is below the floor there, no assumption about how many groups they
really form can rescue it, and you never have to defend a guess.

The middle rows use random assignment, which is a *lower* bound on the damage: items that genuinely
share a group resemble each other more than randomly paired ones do. They show the direction; the
last row carries the argument.

## 7. Is your effect carried by a single group?

A bootstrap over items, or even over training runs, cannot answer this when one side of your
comparison is a **fixed** subset. There is no "which groups are in the subset" variability to
resample, so the interval comes out confident around a number that one group may be carrying.

```python
from groupeval import leave_one_group_out

r = leave_one_group_out(dice, lesion_of_frame, mask=is_in_the_duplicated_slice)
print(f"{r['full']:+.4f} overall, sign stable: {r['sign_stable']}")
for name, d in sorted(r["without"].items(), key=lambda kv: kv[1]):
    print(f"  without {name}: {d:+.4f}")
```

```
+0.1330 overall, sign stable: False
  without s5: -0.1363        <- one group, and the sign flips
  without s4: +0.1761
  without s6: +0.1815
```

This is a real result from the benchmark this package came from: a contrast of **+0.133** with a
bootstrap interval of **[+0.128, +0.138]** reversed to **−0.136** when a single group was removed,
because 82% of the subset was that one group. `largest_share_fraction` reports that concentration
directly, and it is usually the explanation when `sign_stable` is False.

Run it on any comparison where one side is fixed by the dataset rather than sampled by you.
