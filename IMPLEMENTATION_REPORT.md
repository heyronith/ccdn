# Implementation report

Cycle: 2

Base commit: `1815788ebb591775e2d97b7072b7876415e0f135`
Final commit: branch HEAD (SHA reported in handoff)
Tests: `.venv/bin/python -m pytest -q` — 25 passed, 0 failed

Real MNIST verified: YES — torchvision MNIST, 60,000 train / 10,000 test; scientific configs abort on load error and use `allow_synthetic_fallback: false`.

Chosen LR: 0.10 shared SGD learning rate, selected on seed 11 from 0.01, 0.03, 0.10 using mean five-permutation end accuracy averaged across static dense and sparse.

Preflight: all seven models succeeded on real MNIST (seed 11, 5 permutations × 20 batches). All metrics finite, sparse counts matched, RigL and CCDN-0A rewired. Static dense mean end accuracy was 0.519.

Pilot: all 21 runs completed on real MNIST (7 models × seeds 101, 202, 303; 100 permutations × 50 batches; batch 128). Models: static_dense, continual_backprop, static_sparse, selective_reset, rigl, continual_backprop_sparse, ccdn_0a.

Results:
- Raw runs: `results/cycle2_lr_calibration/`, `results/cycle2_preflight/`, `results/cycle2_pilot/`.
- Committed review tables: `review_artifacts/cycle_2/` (calibration, 7 preflight summaries, 21 pilot summaries, 2,100 per-permutation rows, paired sparse differences, resource comparison, manifest).
- Cycle 1 synthetic regression smoke: 10/10 runs succeeded after Cycle 2 changes.

Warnings:
- This is a pilot experiment, not definitive evidence. Dense methods are contextual references, not active-parameter-matched controls. Paired values are raw for three seeds; no significance claims are made. No CCDN success claim is made.
- CUDA and MPS were unavailable; all runs used CPU.
- Raw checkpoints and the MNIST dataset remain gitignored and are not committed.

Ready for review: YES
