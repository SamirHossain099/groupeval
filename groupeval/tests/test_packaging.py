"""Packaging and documentation checks.

Brief §8 asks for MIT, PyPI, CI and docs. These verify the parts that can be checked mechanically,
including that the docs' code examples actually import what they claim: a README that drifts from
the API is worse than no README, because it is confidently wrong.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "groupeval")


def read(*parts):
    p = os.path.join(*parts)
    if not os.path.exists(p):
        pytest.skip(f"{p} absent")
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def test_every_public_name_is_importable_and_exported():
    import groupeval
    for name in groupeval.__all__:
        assert hasattr(groupeval, name), name
    # nothing exported that is not in __all__ by accident
    public = {n for n in dir(groupeval) if not n.startswith("_")}
    # submodules are legitimately reachable; `tests` appears only once pytest has imported it and
    # is not shipped in the wheel (pyproject lists `packages = ["groupeval"]`).
    submodules = {"duplicates", "intervals", "power", "splits", "tests"}
    undeclared = public - set(groupeval.__all__) - submodules
    assert not undeclared, f"public but undeclared in __all__: {undeclared}"
@pytest.mark.parametrize("doc", ["README.md", "EXAMPLE.md"])
def test_everything_the_docs_import_from_groupeval_actually_exists(doc):
    """Parse the `from groupeval import ...` lines and check every name resolves.

    A documentation example that cannot run is worse than none, because it is confidently wrong.
    This catches the realistic failure: renaming a function and forgetting the docs.
    """
    import groupeval
    text = read(PKG, doc)
    imported = set()
    for line in text.splitlines():
        m = re.match(r"\s*from groupeval(?:\.\w+)? import (.+)", line)
        if m:
            imported |= {n.strip() for n in m.group(1).split(",") if n.strip()}
    assert imported, f"{doc} should demonstrate the public API"
    submodules = ("duplicates", "intervals", "power", "splits")
    for name in sorted(imported):
        found = hasattr(groupeval, name) or any(
            hasattr(getattr(groupeval, mod), name) for mod in submodules)
        assert found, f"{doc} imports groupeval.{name}, which does not exist"



def test_worked_example_covers_every_module():
    text = read(PKG, "EXAMPLE.md")
    for name in ("audit_duplicates", "multi_seed_splits", "cluster_ci",
                 "paired_group_test", "decompose_variance", "required_groups"):
        assert name in text, f"EXAMPLE.md does not demonstrate {name}"


def test_citation_metadata_exists_and_matches_the_version():
    import groupeval
    cff = read(ROOT, "CITATION.cff")
    assert f"version: {groupeval.__version__}" in cff
    assert "license: MIT" in cff
    # the package docstring points at this file; the reference must resolve
    assert "CITATION.cff" in groupeval.__doc__


def test_pyproject_declares_mit_and_a_minimal_dependency_set():
    text = read(ROOT, "pyproject.toml")
    assert 'name = "groupeval"' in text
    assert "MIT" in text
    assert 'dependencies = ["numpy' in text, "the package must stay numpy-only"
    assert "torch" not in text and "segmentation" not in text


def test_ci_runs_the_package_suite_on_several_pythons():
    text = read(ROOT, ".github", "workflows", "ci.yml")
    assert "pytest groupeval/tests" in text
    assert "ruff check groupeval" in text
    assert text.count('"3.') >= 2, "CI should cover more than one Python version"


def test_version_is_consistent_between_package_and_pyproject():
    import groupeval
    assert f'version = "{groupeval.__version__}"' in read(ROOT, "pyproject.toml")


# --------------------------------------------------------------- determinism across processes
def test_the_public_api_returns_the_same_numbers_in_any_process():
    """Two interpreters with different PYTHONHASHSEED must agree exactly.

    This is not hypothetical caution. The project this package was extracted from seeded a
    bootstrap with `hash(name)`, and because Python randomises string hashing per process, the same
    command returned different confidence intervals on different runs. A statistics package that
    does that is worse than useless, because the drift is invisible in any single run.
    """
    import subprocess
    import sys
    script = (
        "import numpy as np;"
        "from groupeval import cluster_ci, paired_group_test, decompose_variance,"
        " independence_sensitivity, leave_one_group_out;"
        "rng=np.random.default_rng(0);"
        "g=[f'g{i//5}' for i in range(60)];"
        "v=list(rng.normal(0.8,0.05,60));"
        "w=list(rng.normal(0.8,0.05,60));"
        "s={r:{f'g{i}':0.8+0.01*r+0.02*i for i in range(12)} for r in range(6)};"
        "print(round(cluster_ci(v,g).lo,12), round(cluster_ci(v,g).hi,12));"
        "print(round(paired_group_test(v,w,g).lo,12));"
        "print(round(decompose_variance(s).floor,12));"
        "print([round(r['floor'],12) for r in"
        " independence_sensitivity(s,[f'g{i}' for i in range(6)],counts=(3,),n_rep=5)]);"
        "print(round(leave_one_group_out(v,g,[i<30 for i in range(60)])['full'],12))"
    )
    outputs = set()
    for hs in ("0", "12345", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        outputs.add(r.stdout)
    assert len(outputs) == 1, (
        "groupeval returns different numbers depending on PYTHONHASHSEED:\n"
        + "\n---\n".join(sorted(outputs)))


def test_no_groupeval_module_seeds_a_generator_from_the_builtin_hash():
    import re
    offenders = []
    for name in sorted(os.listdir(PKG)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(PKG, name), encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if re.search(r"seed\s*=\s*hash\(|default_rng\(\s*hash\(", line):
                    offenders.append(f"{name}:{i}")
    assert not offenders, f"builtin hash() used as an RNG seed: {offenders}"


def test_ci_does_not_reference_files_that_do_not_exist():
    """The workflow explains itself by pointing at tests/README; that file has to be there."""
    text = read(ROOT, ".github", "workflows", "ci.yml")
    for m in re.finditer(r"(?:see |in )((?:tests|groupeval)/[\w./-]+)", text):
        ref = m.group(1).rstrip(".,;:")        # the regex otherwise swallows sentence punctuation
        target = os.path.join(ROOT, ref)
        assert os.path.exists(target) or os.path.exists(target + ".md"), (
            f"ci.yml references {ref}, which does not exist")


def test_the_package_source_url_is_still_flagged_as_a_placeholder():
    """`OWNER` is not a GitHub account. Publishing with it would ship a dead link, so the marker
    must survive until someone deliberately sets a real URL."""
    text = read(ROOT, "pyproject.toml")
    if "OWNER" in text:
        assert "PLACEHOLDER" in text, "the placeholder URL is no longer marked as one"
    else:
        assert "github.com/" in text, "a Source URL should be set once it is no longer a placeholder"


def test_the_python_floor_matches_what_the_ci_matrix_tests():
    """requires-python and the CI matrix must agree, or the floor is untested."""
    proj = read(ROOT, "pyproject.toml")
    ci = read(ROOT, ".github", "workflows", "ci.yml")
    m = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', proj)
    assert m, "requires-python not declared"
    assert f'"{m.group(1)}"' in ci, f"CI does not test the declared floor {m.group(1)}"


def test_the_licence_file_that_pyproject_points_at_actually_exists():
    """`license = { file = "LICENSE" }` with no LICENSE means the build fails and every 'MIT'
    claim in the README, CITATION.cff and the package docstring is unbacked."""
    proj = read(ROOT, "pyproject.toml")
    m = re.search(r'license\s*=\s*\{\s*file\s*=\s*"([^"]+)"', proj)
    assert m, "pyproject should point at a licence file"
    path = os.path.join(ROOT, m.group(1))
    assert os.path.exists(path), f"pyproject references {m.group(1)}, which does not exist"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "MIT License" in text, "the licence file is not the MIT licence the metadata claims"
    assert "Samir Hossain" in text, "the copyright holder is not named"


def test_every_place_that_claims_MIT_agrees():
    """Four files assert the licence. They must not be able to disagree."""
    import groupeval
    assert "MIT" in read(ROOT, "pyproject.toml")
    assert "license: MIT" in read(ROOT, "CITATION.cff")
    assert "MIT" in read(PKG, "README.md")
    assert "MIT" in groupeval.__doc__


def test_build_artifacts_and_data_are_ignored():
    """A repo that would commit 804 MB of archives or a stale egg-info is not ready to be one."""
    path = os.path.join(ROOT, ".gitignore")
    assert os.path.exists(path), "no .gitignore; publishing this tree would commit data/ and .venv/"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in ("build/", "*.egg-info/", "__pycache__/", ".venv/", "data/", "results/study/"):
        assert pattern in text, f".gitignore does not cover {pattern}"


def test_nothing_the_package_ships_is_ignored():
    """The inverse mistake: ignoring something the wheel needs. groupeval/ must not be excluded.

    `src/` was on this list while the published repository was the whole audit project. It is not
    any more: the repository is the package, and the audit that the package was extracted from is
    a separate deliverable that `.gitignore` now excludes on purpose. Listing it here asserted the
    old layout and failed the moment the new one was correct.
    """
    with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    for needed in ("groupeval/", "groupeval", "LICENSE", "pyproject.toml", "README.md"):
        assert needed not in lines, f".gitignore excludes {needed}, which is part of the artifact"


@pytest.mark.parametrize("where,doc", [(PKG, "README.md"), (PKG, "EXAMPLE.md"),
                                       (ROOT, "README.md")],
                         ids=["pkg-README", "pkg-EXAMPLE", "root-README"])
def test_every_documented_call_matches_the_real_signature(where, doc):
    """Names existing is not enough: a call with an argument the function does not take fails the
    moment a reader pastes it. The examples use placeholder variables so they cannot be executed,
    but their call signatures can be bound against the real ones.

    The ROOT README is checked too. It carries its own copy of the quickstart, and it is the first
    thing a visitor sees, so an example that drifts from the API there is the most expensive one to
    get wrong: it was outside this test until the repository was narrowed to the package.
    """
    import ast
    import inspect

    import groupeval
    text = read(where, doc)
    public = {n: getattr(groupeval, n) for n in groupeval.__all__
              if callable(getattr(groupeval, n))}
    checked = 0
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:                       # a doc example must at least parse
            raise AssertionError(f"{doc}: example does not parse: {exc}") from exc
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            fn = public.get(node.func.id)
            if fn is None:
                continue
            kwargs = {k.arg for k in node.keywords if k.arg}
            args = ["_"] * len(node.args)
            try:
                inspect.signature(fn).bind_partial(*args, **{k: None for k in kwargs})
            except TypeError as exc:
                raise AssertionError(
                    f"{doc}: `{node.func.id}(...)` does not match its signature "
                    f"{inspect.signature(fn)}, {exc}") from exc
            checked += 1
    assert checked, f"{doc}: no calls to the public API were checked"


def test_the_sdist_contains_nothing_outside_the_package():
    """A source distribution built from the project root must not sweep in its neighbours.

    setuptools auto-includes a top-level `tests/` directory, and this package lives inside the
    working tree of the audit it was extracted from. `python -m build` from the project root
    therefore put the audit's whole 41-file suite into the tarball, including 63 KB of assertions
    about an unpublished manuscript, and grew it from 19 KB to 128 KB. Building from a clean
    `git archive` avoided it, but only if you remembered to; `MANIFEST.in` now prunes it, and this
    checks the result rather than the intention.

    Skips when nothing has been built, so it costs nothing in CI.
    """
    import glob
    import tarfile

    tarballs = glob.glob(os.path.join(ROOT, "dist", "*.tar.gz"))
    if not tarballs:
        pytest.skip("no sdist built")
    newest = max(tarballs, key=os.path.getmtime)
    with tarfile.open(newest) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]

    assert names, f"{os.path.basename(newest)} is empty"
    root = names[0].split("/")[0]
    allowed_top = {"LICENSE", "README.md", "CITATION.cff", "MANIFEST.in", "pyproject.toml",
                   "setup.cfg", "PKG-INFO"}
    strays = []
    for n in names:
        rel = n[len(root) + 1:]
        if rel.startswith(("groupeval/", "groupeval.egg-info/")) or rel in allowed_top:
            continue
        strays.append(rel)
    assert not strays, f"the sdist carries files from outside the package: {sorted(strays)[:10]}"
