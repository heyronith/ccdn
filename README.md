# CCDN experiment foundation

Conserved-Capacity Developmental Networks (CCDN) is a research project studying loss of plasticity during long-horizon continual learning. **Cycle 1 contains only experiment infrastructure and baseline algorithms; CCDN itself is not implemented, and no research result is claimed.**

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

MNIST downloads through torchvision on the first real-data run. The training command falls back to seeded synthetic MNIST-shaped examples if loading MNIST fails and records that fact in `summary.json`. Synthetic runs validate plumbing only; they are not MNIST results.

## Run

```bash
pytest
python -m ccdn.experiments.train --config configs/pmnist/static_dense.yaml
python -m ccdn.experiments.run_matrix --suite configs/suites/smoke.yaml
python -m ccdn.experiments.summarize results/smoke
```

Every run gets a timestamped unique directory below `results/<suite>/<model>/seed_<seed>/`, with resolved config, per-evaluation `metrics.csv`, adaptation AUC, summary, and a resumable checkpoint. A run never silently overwrites an earlier directory.

## Layout

- `streams/`: evaluator-owned online permuted MNIST stream.
- `models/`: dense and explicit-mask MLPs.
- `baselines/`: algorithm policies, using global update cadence and internal signals.
- `metrics/`: continual-learning, plasticity, and resource metrics.
- `experiments/`: generic training and model-by-seed matrix.
- `utils/`: configuration, RNG, checkpoint, and logging utilities.

## Reproducibility and limitations

One seed utility covers Python, NumPy, PyTorch CPU, and CUDA. Permutations derive from `seed + permutation_index`; learners receive only `(x, y)`, while the runner keeps permutation metadata for evaluation. Checkpoints include model, algorithm, optimizer, step, config, and RNG state. Device defaults to CUDA when available and otherwise CPU.

Sparse layers use dense tensors and masks: logical active capacity is reported separately from allocated tensor parameters and dense execution cost. Continual Backprop is a lightweight documented reproduction using activation-times-sensitivity utility, maturity, and cadence-based unit replacement; it is not an exact paper reproduction. RigL uses a practical dense gradient estimate for inactive candidates. Adaptation AUC is measured from evaluation points within each permutation. Compute and evaluation are intentionally small for the smoke suite; extended sweeps and larger diagnostic studies belong to later cycles.

## Cycle 2: CCDN-0A pilot

Cycle 1 is accepted. Cycle 2 adds **CCDN-0A**, a rewiring-only method that combines ordinary gradient learning, EMA connection utility, utility-based pruning, and gradient-guided regrowth under a fixed per-layer sparse edge budget. This is the first proposed CCDN mechanism. Cycle 2 also adds a resource-matched sparse adaptation of the lightweight Continual Backprop control. It does not include consolidation or a Plasticity Reserve Score (PRS).

The real-MNIST calibration, preflight, and exploratory three-seed pilot are configured in `configs/cycle2/` and `configs/suites/cycle2_*.yaml`. Scientific runs set `synthetic: false` and `allow_synthetic_fallback: false`; a data-loading failure aborts those runs. The development calibration selects one shared LR across static dense and static sparse. Pilot comparisons label the five active-parameter-matched sparse methods separately from contextual dense references.

Cycle 2 is an exploratory pilot, not definitive evidence. It does not establish that CCDN-0A works, and no consolidation, PRS, or future-trainability mechanism is implemented.

## Cycle 2V: official baseline references

Cycle 1 is accepted. Cycle 2V pins official Continual Backprop / Loss of Plasticity (`shibhansh/loss-of-plasticity`, `a6b79580d85f3025bdb601566d3627c5f489f13b`) and RigL (`google-research/rigl`, `d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9`) sources, runs compact official-code execution smokes, and adds separately named PyTorch reference ports checked against upstream tensor traces. The existing Cycle 2 `continual_backprop` and `rigl` methods remain local lightweight approximations; use `continual_backprop_reference` and `rigl_reference` for the validated reference ports. RigL's smoke needed a TensorFlow 2.15 legacy-SGD runtime compatibility shim; the upstream checkout remains unchanged.

Cycle 2V does not run CCDN performance comparisons or establish that CCDN works. These artifacts validate reference implementations and execution paths only. See `official_baselines/README.md` and `review_artifacts/cycle_2v/README.md` for source pins, evidence, and limitations. To fetch and check the source snapshots, run `python scripts/fetch_official_baselines.py`; the downloaded trees and isolated runtimes are ignored by git. After installing the validation environments, `python -m ccdn.validation.official_baselines` repeats parity checks. Pass `--full` to rerun the small real-MNIST upstream smokes.

## Cycle 2B: online loss-of-plasticity reproduction

Cycle 2B adds a separate published-style single-example Online Permuted MNIST runner with task-boundary checkpoints and no task signals to the learner. It uses the pinned official Backprop implementation for runner parity and the validated `continual_backprop_reference` only after the preregistered Backprop failure gate passes. The 150-task real-MNIST run completed all 9,000,000 updates. In this single-seed pilot, Backprop's peak 20-task online accuracy was 0.9227 and its final 20-task accuracy was 0.8718 (drop 0.0509); the fixed descriptive gate passed. The matched CBP reference's drop was 0.0007 and its final 20-task accuracy was 0.9279. These are measurements for reviewer inspection, not significance claims. **CCDN has not yet been tested in the validated failure regime.**

Cycle 2B also measures the sparse initialization confound: active-fan-in Kaiming produced higher early adaptation AUC than historical masked dense-fan-in initialization in the small sanity check. See `review_artifacts/cycle_2b/` for the task curves, diagnostics, parity evidence, paired metrics, and exact protocol.
