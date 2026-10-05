# Implementation report

Cycle: 1

Git commit: HEAD (final SHA reported in the handoff)
Python version: 3.12.13
PyTorch version: 2.14.1

Files/modules added:
- `ccdn/`: dense/sparse MLPs, five baseline policies, permuted-MNIST stream, metrics, experiment runners, reproducibility, checkpointing, logging, config.
- `configs/pmnist/`: one YAML config per baseline.
- `configs/suites/smoke.yaml`: five models × two seeds, 20 permutations.
- `tests/`: 13 tests covering data isolation, baselines, reproducibility, metrics, checkpointing, and resource accounting.
- `README.md`, `pyproject.toml`, `.gitignore`.

Tests:
- command: `.venv/bin/python -m pytest -q`
- passed: 13
- failed: 0

Commands executed:
- `uv pip install --python .venv/bin/python 'torch>=2.1' 'torchvision>=0.16' 'PyYAML>=6' 'pytest>=7.4' 'pandas>=2.0'`
- `uv pip install --python .venv/bin/python --no-build-isolation -e .`
- `.venv/bin/python -m ccdn.experiments.run_matrix --suite configs/suites/smoke.yaml`

Smoke experiments:
- models: static_dense, static_sparse, selective_reset, continual_backprop, rigl
- seeds: 1, 2
- permutations: 20 per run
- data: seeded synthetic MNIST-shaped examples; real MNIST was not run, and no scientific MNIST result is claimed

Result locations:
- `results/smoke/<model>/seed_<seed>/<timestamp>_<id>/`
- Each run contains resolved config, metrics, per-permutation adaptation AUC, summary, and checkpoint.

Key sanity observations:
- All 10 runs completed with finite evaluation metrics.
- Sparse baselines report 54,282 logical active parameters against 269,322 dense tensor parameters for the configured network.
- RigL preserved the active parameter count across recorded evaluations.
- Learner training callbacks receive no permutation ID or boundary signal; boundary metadata stays in the evaluator.

Known limitations / deviations:
- Continual Backprop is a lightweight activation-times-sensitivity utility reproduction, not an exact paper reproduction.
- RigL and static sparse models use dense masked tensor execution; inactive-candidate scores use dense layer-input × backpropagated-delta estimates.
- Smoke uses two batches per permutation and one eval batch per point; larger scientific runs need longer schedules.

Warnings:
- Synthetic smoke results validate integration only. Real-data download/training remains unverified.

Ready for review: YES
