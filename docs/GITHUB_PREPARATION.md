# Preparing the Stage-1 GitHub publication

The publication contains the Stage-1 reference implementation, its tests and
documentation. Use its explicit file manifest to prepare a reviewable commit.

The explicit [allowlist](../configs/stage1_publication.json) contains the
source/test paths, documentation, example, CI workflow and exporter. New
publication files require an explicit change to this list.
It also includes benchmark configurations, runners and the recorded synthetic
comparison under `benchmarks/reference/`, with compressed evidence and a frozen
source archive.

## Build a reviewable tree

Run from the repository root:

```bash
python scripts/prepare_stage1_release.py --check
python scripts/prepare_stage1_release.py
```

The default output is `build/stage1/`, ignored by Git. An existing destination is
refused. For another snapshot, select a new destination:

```bash
python scripts/prepare_stage1_release.py --output build/stage1-review-2
```

The exporter validates repository-relative source paths and resolves them within
the repository. It reads all allowed inputs before writing to a new destination,
preserving source files and the Git index. Source directories and existing output
directories are protected. If a write fails, the partial destination remains
available for inspection; use a fresh destination after resolving the cause.

`stage1-manifest.json` records each exported path, byte count, SHA256 and source
provenance, plus the exact allowlist hash. The script prints the SHA256 of that
generated manifest. File identity and scientific verification are recorded
separately, with test evidence in [STAGE1_VALIDATION.md](STAGE1_VALIDATION.md).

Python, Markdown, TOML, JSON, YAML, `.gitignore` and `.gitattributes` are exported
with LF line endings, matching the checkout rules in `.gitattributes`. Their
manifest hashes cover the exported bytes. When transport normalization changes
bytes, `input_sha256` retains the pre-normalization hash; `text_normalization`
records the LF policy. A projected published file also retains its raw original
`published_blob_sha256`. Only newline bytes are normalized during this step.

The published allowlist reads the committed Stage-1 files directly. This makes
preparation reproducible from a fresh clone. When assembling a release from
versioned inputs, the generated manifest retains their source revision, original
blob hash and named projection. The exported configuration selects the delivered
files as its inputs, preserving the original configuration hash in the manifest.

## Validate the actual publication

From the exported tree, use a fresh environment or the existing project
interpreter with explicit source resolution. The clean-install workflow is:

```bash
cd build/stage1
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python examples/stage1_quickstart.py
python scripts/run_synthetic_benchmark.py --config configs/synthetic_benchmark_smoke.json --output results/ci-benchmark
```

The GitHub workflow runs the regression, example and bounded benchmark with Python 3.13 and 3.14
using HiGHS. Optional Gurobi integration skips when it is unavailable. The
regression must run against the actual delivered tree. Record actual results in
[STAGE1_VALIDATION.md](STAGE1_VALIDATION.md), then rebuild so the final
manifest also covers the completed verification record.

Workflow action versions follow the maintained
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) usage documentation.
The hosted job requires only repository read access.

## Review before a commit

The export has no `.git` directory. To preserve existing history, copy its files
into a separate checkout based on the intended repository branch and review the
diff there. Stage only the manifest's files plus the generated manifest. For
example, from that separate checkout in PowerShell:

```powershell
$publicationFiles = (Get-Content stage1-manifest.json -Raw | ConvertFrom-Json).files.path
git add -- $publicationFiles
git add -- stage1-manifest.json
git diff --cached --check
git diff --cached --stat
```

Review the staged diff against the manifest and check for generated artifacts,
local paths and secrets before committing. This procedure preserves the existing
repository history.

Literal fixtures preserve their original unit policy for regression checks. The
README links the papers directly. Select a reuse license before offering external
users reuse rights.
