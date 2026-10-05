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
