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

## Cycle 2C — Phase A protocol preparation

Cycle: 2C Phase A

Base commit: `0386a9076d755d5fb37f84dc926bad68269f03ef`
Final commit: reported after commit
Branch: `cycle-2c-ccdn-failure-regime`

Tests: `.venv/bin/python -m pytest -q` — 60 passed, 0 failed. Existing synthetic smoke matrix: 10/10 runs completed.

Frozen methods: `static_sparse`, `selective_reset`, `rigl_reference`, `ccdn_0a`. All use 784–336–336–336–10, density 0.20, active-fan-in Kaiming, SGD 0.003, and paired seed-101 initialization. CCDN-0A remains rewiring-only. Selective Reset adds the per-layer selection mode while retaining the historical global default. Structural cadence is 8,192 completed updates; RigL Reference uses `begin_step=8191` to align its pre-increment schedule.

Real MNIST verified: YES, 60,000 training examples, 784 inputs, labels 0–9, no synthetic fallback. Exact task-stream SHA256: `1f2ddb0daa90715192118466b7088fd8d3ea78f44e8647af6bca0a686393f844`.

Preflight: PASS — four methods × 8,192 real examples. Initial weights/biases/masks matched. The state remained identical through update 8,191. At update 8,192, Selective Reset reset weights without mask changes; RigL Reference and CCDN-0A changed topology while conserving every layer's edge count. Check `review_artifacts/cycle_2c_preflight/`.

Long scientific run: **NOT EXECUTED.** No 150-task / 9,000,000-update experiment or dynamic scientific comparison was launched. No CCDN outcome is claimed.

Warnings: Phase A preflight is a software integration check and is not scientific evidence. The full static-sparse gate and subsequent comparisons require independent review and the normal terminal command.

Ready for review: YES — protocol-ready Phase A only; not approval to execute the long run.

## Cycle 2C artifact collection

Collection-only commit: `8467b07db9d87512ad45882829b0f36a269c8e5c` on `cycle-2c-ccdn-failure-regime`. The verified source run is `seed101_5b48752c_019e3067`; Static Sparse completed 150 tasks / 9,000,000 updates and the preregistered 150-task gate failed (drop `0.02406749999999991`). The checkpoint was not committed. The collected package is `review_artifacts/cycle_2c/`.

## Cycle 2D Phase A — sparse lifetime calibration

Base commit: `8467b07db9d87512ad45882829b0f36a269c8e5c`
Final commit: reported in handoff
Branch: `cycle-2d-sparse-lifetime-calibration`

The frozen candidates are `[200, 250, 300, 400, 600, 800]`. A horizon is selected only when it and its immediately following candidate both pass the preregistered drop/trend gate. The protocol continues from Cycle 2C task index 149 and changes only maximum lifetime. No Cycle 2D comparison experiment was run or inspected; CCDN remains unseen.

Cycle 2C source checkpoint SHA-256: `37492b00e1aef9db440be3374d6b6e79a816ee777a17afcc9e3e5e1bc2ecf779`; source protocol hash: `019e306798b19183d1b4ee32a7c3ee200ad176f100c55900c97a275eb8f01cb1`. Prefix/task hashes and state inheritance evidence are in `review_artifacts/cycle_2d_preflight/`.

Tests: `.venv/bin/python -m pytest -q` — 129 passed, 0 failed. Real-MNIST preflight: PASS. Model, optimizer, algorithm, RNG, masks, checkpoint source bytes, and historical task/diagnostic CSVs matched exactly; Task 151 is the next task. Training updates during preflight: 0. Cycle 2D protocol hash: `0ff18ca9429a607171532694b748f95de547914e551f51bbbec3d8c3131fa723`.

Long Cycle 2D continuation: **NOT EXECUTED.** Do not launch `./scripts/run_cycle2d.sh` until the reviewer approves the final protocol-ready SHA.

## Cycle 2C — Phase A final hardening

Base commit: `6a0b8b4de8c47e73ea310cc450b940de338900ba`
Final commit: reported in handoff
Branch: `cycle-2c-ccdn-failure-regime`

Tests: `.venv/bin/python -m pytest -q` — 72 passed, 0 failed. Synthetic regression: 10/10.

Hardening: rolling checkpoint recovery prefers valid latest, falls back to valid previous, and refuses invalid/missing state; active-mask overlap counts retained initially active edges; terminal execution requires an exact reviewer-approved full SHA; task-boundary heartbeats report and assert expected/current edge budgets, finite state, and checkpoint health; status/heartbeat JSON writes are atomic; finite checks occur after each completed 4,096th update.

Real-MNIST preflight: PASS — all four sparse methods, 8,192 examples each; identical paired initialization and state through update 8,191; first intervention at 8,192; 99,533 logical active parameters; finite state, checkpoint roundtrip, and edge conservation passed. Synthetic fallback was false.

Long scientific run: **NOT EXECUTED.** No 150-task method was started. The approved terminal command is `./scripts/run_cycle2c.sh <FULL_SHA>` after independent review.
