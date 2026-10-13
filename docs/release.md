# v0.2.0 — Validated A100 release

This release incorporates the successful full-forward A100 run, preserves the
earlier transport checkpoint, and adds a portable results explorer, README figure,
recomputed evidence summary, integrity checks, and reproduction notebooks.
All 22 measured package source files are unchanged from the validated run.

## Open the demo

Open `demo/index.html` directly in a browser. No server, external assets, or GPU
is required. The demo filters workloads, compares stages and median/p95, displays
expert loads, and downloads the original full-forward JSON. It visualizes recorded
benchmarks; run the CLI/API to execute the engine.

## Verify and rebuild

```bash
python -m pip install -e '.[dev,notebook,plot]'
python scripts/verify_release.py
python -m pytest -q
python -m ruff check src tests scripts
python scripts/finalize_assets.py
python scripts/build_colab.py
python scripts/build_colab.py --extras-only
python scripts/build_release.py
```

GPU tests skip on CPU hosts. The recorded A100 test results remain in
`benchmarks/a100-block/run.log`. `docs/release_validation.json` records the
separate release checks. The old `docs/local_validation.json` is a historical
pre-GPU checkpoint.

`build_release.py` excludes caches, build output, Git internals, and scratch
`runs/`; archived evidence is under `benchmarks/`. It sorts entries, fixes ZIP
metadata, and emits a SHA-256 sidecar. Given the same input bytes and Python/zlib
runtime, rebuilding produces the same archive bytes. Its fixed ZIP timestamp is
packaging metadata; it does not alter Git history or benchmark run timestamps.

The full reproduction notebook embeds the checkout. The optional notebook runs
quick CUDA Graph comparisons first and the larger model-shaped preset second.
Graph capture failures and preset failures are exported as diagnostics. Both optional paths completed successfully in the additional A100 run.
Graph replay passed 24 checks and measured 1.74× median graph-to-graph ratio;
the larger preset passed 6 oracle comparisons and measured 1.06× median eager
ratio. Those raw records and their separate log are archived with the release.

## Publish a new GitHub repository

Target name: `AarushRathore1729/cuda-moe-engine`.

With GitHub CLI authenticated on your machine, run from the extracted folder:

```bash
git init -b main
git add .
git commit -m "Release validated A100 CUDA MoE forward engine"
gh repo create AarushRathore1729/cuda-moe-engine --public --source=. --remote=origin --push
gh release create v0.2.0 dist/cuda-moe-engine-v0.2.0.zip dist/cuda-moe-engine-v0.2.0.zip.sha256 --title "v0.2.0: validated A100 forward engine" --notes-file docs/resume.md
```

Build the archive before the release command. If the repository already exists,
use its clone/remote instead of creating a second repository. A web-created empty
repository can also receive the finalized files through the connected GitHub tools.
GitHub Pages is optional: the demo works directly from the release ZIP.

## Measured scope

3.17× is synthetic single-block eager forward versus the included PyTorch
variant with shared Triton expert GEMMs. 6.36× and 72.7% belong to standalone
transport. CUDA-event timing can include CPU launch gaps. Graph and larger-preset results are reported separately. No full-model,
multi-GPU, training, or production-library superiority claim is made.
