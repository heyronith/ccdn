# Implementation report

Cycle: 1 — final review fixes

Git commit: branch HEAD (SHA reported in handoff)
Python version: 3.12.13
PyTorch version: 2.14.1

Files/modules added or updated:
- `ccdn/models/initialization.py`: shared PyTorch Linear-equivalent uniform initializer using full-layer fan-in.
- `ccdn/models/dense_mlp.py`, `ccdn/models/sparse_mlp.py`: scale-correct unit/weight reset methods.
- `ccdn/baselines/`: Selective Reset and Continual Backprop now use layer-consistent initialization; reset/replacement/rewire positions are retained in algorithm state.
- `tests/`: explicit initialization-bound and SGD momentum-clearing checks; boundary-free hook coverage for all five baselines.
- `review_artifacts/cycle_1/`: fresh smoke configs, summaries, adaptation curves, and per-evaluation metrics.

Tests:
- command: `python -m pytest -q`
- passed: 15
- failed: 0

Commands executed:
- `python -m pytest -q`
- `python -m ccdn.experiments.run_matrix --suite configs/suites/smoke.yaml`

Smoke experiments:
- models: static_dense, static_sparse, selective_reset, continual_backprop, rigl
- seeds: 1, 2; permutations: 20 per run
- all 10 runs succeeded on fresh seeded synthetic MNIST-shaped data
- each full per-run `metrics.csv` is included; aggregate files are also provided

Result locations:
- raw runs: `results/smoke/<model>/seed_<seed>/<timestamp>_<id>/`
- committed review bundle: `review_artifacts/cycle_1/`

Key sanity observations:
- Dense unit replacement and Selective Reset use uniform bounds derived from each full layer's `in_features`.
- SGD momentum entries are verified cleared for reset weights, Continual Backprop incoming rows/outgoing columns, and RigL pruned/grown weights.
- All five baselines' training hooks reject task/permutation/boundary arguments; learner batches remain `(x, y)`.
- Smoke summaries and evaluation metrics are finite; RigL's logical active parameter count is conserved.

Known limitations / deviations:
- Continual Backprop remains a lightweight reproduction, and sparse execution still uses dense masked tensors.

Warnings:
- Synthetic integration smoke test only. These are not scientific MNIST results. Real-data training was not run.
- No checkpoints or datasets are included in the committed review artifacts.

Ready for review: YES
