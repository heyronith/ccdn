# Implementation report

Cycle: 2V — Official Baseline Acquisition, Execution, and Harness Parity

Base commit: `c663d62f9658e9f6dac1325a408a607ff83ab11a`
Final commit: this report's containing commit (SHA reported in handoff)
Branch: `cycle-2v-official-baselines`

Python: validation environments use Python 3.11.15; repository test environment uses Python 3.12.
PyTorch: CBP environment 2.1.0; repository environment 2.14.1.

Official sources: Continual Backprop `a6b79580d85f3025bdb601566d3627c5f489f13b`; RigL `d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9`. Both fetched trees are pristine and gitignored.

Tests:
- Command: `.venv/bin/python -m pytest -q`
- Passed: 35
- Failed: 0

Validation commands:
- `python scripts/fetch_official_baselines.py` — pinned source fetch and SHA verification.
- `.venv/bin/python -m ccdn.validation.official_baselines` — passed source verification, upstream/local parity probes, and evidence collection.
- `.venv/bin/python -m ccdn.experiments.run_matrix --suite configs/suites/smoke.yaml` — existing Cycle 1 synthetic smoke suite completed all 10 model/seed runs.
- Official CBP PyTorch MNIST loader and `online_expr.py` — real-MNIST execution completed; 6 optimizer updates across two 30,000-example segments.
- Official RigL TensorFlow `train.py` — real-MNIST execution completed; 3 optimizer updates and mask changes in each sparse layer.

Data: real MNIST for both upstream execution smokes; no synthetic fallback. These are framework/execution checks, not CCDN experiments or performance comparisons.

Artifacts: `review_artifacts/cycle_2v/` — provenance, runtime versions, execution summaries, CBP mechanism/parity checks, RigL parity, validation summary, and legacy-vs-reference notes. External source clones, environments, datasets, and raw logs remain gitignored.

Legacy-vs-reference differences: CBP utility, bias correction, maturity, replacement accumulation, and reset ordering differ from the lightweight Cycle 2 method. RigL schedule/interface and gradient path differ from Cycle 2's local implementation; details are in `review_artifacts/cycle_2v/legacy_vs_reference.md`.

Validation artifacts: `review_artifacts/cycle_2v/`.

Known limitations:
- RigL execution is an environment-patched official execution: an external TensorFlow 2.15 legacy-SGD compatibility wrapper supplies the API expected by the pinned source, and a runtime alias restores a removed TF1 assertion symbol for parity checks. The pinned upstream checkout is unchanged. Its official smoke uses a deliberately small 32×32 MLP, three updates, and initial one-shot 80% sparsity to exercise rewiring.
- CBP smoke uses a 100-unit, three-hidden-layer network, two 30,000-example segments, and six batches of 10,000. Its low accuracy is only execution evidence.
- Reference parity is validated with focused traces, not claimed for every upstream mode/configuration. RigL's local candidate-gradient path uses the existing practical dense approximation.
- No CCDN performance comparison was run. No performance or scientific claim is made.

Ready for scientific comparison: YES
