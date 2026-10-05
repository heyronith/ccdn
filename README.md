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
