# Implementation report

Cycle: 2B

Base commit: `700cce4b61878a9fbefbfa9fc5f2030a70457e9c`
Final commit: this report's containing commit (SHA reported in handoff)
Branch: `cycle-2b-plasticity-reproduction`

Tests: `.venv/bin/python -m pytest -q` — 47 passed, 0 failed. Synthetic regression matrix: 10/10 model/seed runs passed.

RigL dense candidate-gradient equivalence: PASS — reconstructed gradients match autograd at every dense effective-weight position in input-to-hidden, hidden-to-hidden, and hidden-to-output layers (`rtol=atol=1e-12`, float64).

Official Backprop runner parity: PASS — pinned `shibhansh/loss-of-plasticity` commit `a6b79580d85f3025bdb601566d3627c5f489f13b`; 9 one-example updates matched official `Backprop` predictions-before-update, loss, weights, and biases; maximum absolute errors were zero.

Online PMNIST reproduction: 150/150 tasks, 9,000,000/9,000,000 updates on real MNIST. Seed 101, architecture 784–100–100–100–10, published Kaiming initialization, SGD 0.003. Runtime: 1,208 s Backprop; 2,106 s CBP.

Backprop gate: peak 20-task accuracy 0.922660 (window tasks 0–19; peak task 19); final 20-task accuracy 0.871757; drop 0.050903. Peak preceded the final region; post-peak and final-50 OLS slopes were both negative. Gate: PASS.

Reference CBP executed: YES — pinned Cycle 2V reference port from `shibhansh/loss-of-plasticity` commit `a6b79580d85f3025bdb601566d3627c5f489f13b`; same sequence hash, initialization, architecture, optimizer, seed, and 9,000,000 updates. Final-20 accuracy 0.927945; drop 0.000706; 26,970 replacements; final dead-unit fraction 0.0 versus Backprop 0.17; final mean effective rank 51.15 versus Backprop 25.22. Measurements only; one seed, no significance claim.

Sparse initialization sanity: real MNIST, seed 11, 5 permutations × 20 batches, density 0.20. Legacy vs active-fan-in Kaiming: initial activation std 0.282442 vs 0.261134; mean adaptation AUC 0.117148 vs 0.318555; mean end accuracy 0.128750 vs 0.500000. This exposes an initialization confound and is not evidence about CCDN.

Artifacts: `review_artifacts/cycle_2b/` (parity, task accuracies, diagnostics, summaries, paired CSV, plots, sparse initialization check). Raw data and checkpoints remain under gitignored `data/` and `results/`.

Warnings: single-seed descriptive pilot. Backprop gate passes narrowly (drop 0.0509 against 0.05 threshold). Reviewer should inspect the committed curves and diagnostics before authorizing CCDN testing. No CCDN mechanism or CCDN experiment was run.

Ready to test CCDN in failure regime: YES — measurements are available for reviewer inspection; CCDN itself was not tested.
