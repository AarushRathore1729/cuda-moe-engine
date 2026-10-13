# Validation status

## Recorded A100 execution — 2026-10-13

The v0.2.0 engine ran on NVIDIA A100-SXM4-40GB with Python 3.13.15,
PyTorch 2.11.0+cu130, and CUDA 13.0. The complete log is retained in
`benchmarks/a100-block/run.log`.

| Check | Recorded result |
|---|---|
| CPU pytest | 81 passed |
| GPU pytest | 39 passed |
| Standalone transport validation | 36 passed, zero failures |
| Quick full-forward oracle comparisons | 24 / 24 passed (8 cases × 3 variants) |
| Smoke full-forward oracle comparisons | 6 / 6 passed |
| Quick maximum absolute error | 0.0078125 |
| Quick maximum relative L2 error | 0.0002142672601621598 |
| Measured source hashes | 22 / 22 match the release |
| Profiler | Trace contains 36 GPU kernel events |

Each comparison uses an independent token-major oracle, rtol=0.03 and atol=0.03,
plus a relative L2 limit of 0.02. Reported errors are measurements, not claims of
bitwise equivalence. Tests cover empty/skewed experts, aligned padding, precision,
odd widths, routing contracts, and buffer behavior.

The quick suite has 120 timing rows (8 cases × 3 implementations × 5 stages).
All raw sample counts, medians, p95 and minima are verified by
`python scripts/verify_release.py`. Smoke adds 30 independently checked rows.
The archived transport source and evidence are also hash-verified.

## Release verification

The final release keeps all 22 measured package source files byte-identical.
Changes after the A100 run add documentation, figures, offline demo, evidence
verification, release packaging, and notebook orchestration. The original 81 CPU
tests passed again in the release workspace with PyTorch 2.11.0+cpu. Additional
recorded-evidence regression tests are summarized in `docs/release_validation.json`.
GPU tests were not rerun in the CPU release workspace; their successful A100
execution is preserved in the log above. `docs/local_validation.json` records the
earlier pre-GPU local checkpoint and remains historical evidence.

## Additional A100 execution — 2026-10-13

`benchmarks/a100-block/optional_run.log` records the successful optional runs.
The package sources match the original 22 hashes exactly.

| Experiment | Checks | Timing rows | Measured ratio |
|---|---|---|---|
| Quick suite with graphs | 24 eager oracle comparisons + 24 replay/eager checks | 144 | Graph-to-graph median 1.735964958×; eager rerun median 3.083333382× |
| Larger synthetic preset | 6 independent-oracle comparisons | 30 | Eager median 1.064692398× |

Every graph variant captured and replayed successfully. The graph ratio range is
1.146214070–2.529411689×. The larger preset uses H=4096, I=14336, E=8, K=2, with
T=128 balanced and T=256 skewed. Maximum absolute error was 0.015625, maximum
relative L2 error was 0.0031671082, and all comparisons passed the documented BF16
thresholds. Larger-preset graph replay was not requested or measured.

The original 3.17× eager checkpoint is preserved separately. These results do not
establish a uniform ratio across shapes or execution modes. No backward pass,
multi-GPU, pretrained-model quality, or production-library comparison is included.
