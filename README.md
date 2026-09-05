# groupeval

**Evaluation that respects the group structure benchmarks actually have.**

Most vision and medical-imaging benchmarks are built from *groups* (video frames of one lesion,
slices of one scan, crops of one slide, images of one patient) and then scored as if every file were
an independent observation. Three things go wrong, and all three are arithmetic rather than
judgement:

| problem | what it looks like | what `groupeval` does |
|---|---|---|
| **duplicates** | the same image in two "independent" test sets, or twice in one | exact pixel audit, no thresholds |
| **splits** | train/test separates files but not groups | group-disjoint splits, leakage asserted |
| **intervals** | confidence intervals computed over files | cluster bootstrap and design effect |
| **resolution** | "not significant", but is that the data or the compute? | variance decomposition and a floor |

## Install

```bash
pip install -e .
```

numpy is the only runtime dependency. Python 3.9+.

## Quickstart

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

**Full documentation is in [`groupeval/README.md`](groupeval/README.md)**, with a worked end-to-end
example in [`groupeval/EXAMPLE.md`](groupeval/EXAMPLE.md).

## Why it exists

The package was extracted from an independent audit of the standard polyp segmentation benchmark,
where every one of these problems turned out to be present at once: a test set whose 60 images were
all duplicates of another test set's, a shipped train/test split that separated files but not
lesions, per-frame intervals on data with an intra-cluster correlation of 0.5, and a resolution
floor below which most published improvements sit.

None of that is specific to polyps, which is why the tools here are dataset-agnostic and take group
labels rather than any particular data format. The audit itself and its manuscript are separate
work; this repository is the reusable part.

## Tests

```bash
python -m pytest groupeval/tests -q
```

The suite includes a test that re-derives the audit's central finding from raw pixels through the
public API alone. It skips when the source archive is absent, which it is here, because those
archives are not ours to redistribute.

## Citing

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT, see [`LICENSE`](LICENSE).
