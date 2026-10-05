# CCDN Cycle 2 review package

> **Pilot experiment — not definitive evidence.**

All calibration, preflight, and pilot model runs summarized here used actual torchvision MNIST with synthetic fallback disabled. The stream contains 60,000 training and 10,000 test examples.

## Protocol

- Shared SGD learning rate: **0.10**, chosen before preflight/pilot by averaging five-permutation end accuracy for `static_dense` and `static_sparse` over candidates 0.01, 0.03, and 0.10.
- Preflight: seed 11, seven methods, five permutations × 20 batches.
- Pilot: seven methods × seeds 101, 202, 303; 100 permutations × 50 batches; batch size 128; hidden sizes [512, 512]; sparse density 0.20.
- Evaluations every 10 updates using 10 test batches. Rewiring/replacement cadence is 64 updates.

## Comparison groups

`static_sparse`, `selective_reset`, `rigl`, `continual_backprop_sparse`, and `ccdn_0a` are the primary resource-matched sparse comparison. Each uses the same seed-specific starting weights, biases, and masks, with 134,769 logical active parameters. `static_dense` and `continual_backprop` are contextual dense references; their active parameter count is not matched to the sparse group.

## Files

- `lr_calibration.csv`: six model/rate calibration runs and selection averages.
- `preflight_summaries.csv`: seven real-MNIST preflight runs.
- `pilot_summaries.csv`: one row per model/seed with lifetime AUC and resource measures.
- `permutation_auc.csv`: adaptation AUC and end accuracy for every permutation.
- `paired_sparse_differences.csv`: raw per-seed differences for mean, early, late, late-minus-early, and slope AUC plus mean end accuracy; CCDN-0A minus each primary sparse comparator. No p-values or significance claims are made.
- `resource_comparison.csv`: mean resource accounting by model and group.
- `manifest.json`: protocol and verification facts.

These results are a signal-finding pilot only. This package makes measurements and does not claim that CCDN-0A works. Raw outputs and checkpoints remain under gitignored `results/`; no datasets, checkpoints, or utility tensors are included here.
