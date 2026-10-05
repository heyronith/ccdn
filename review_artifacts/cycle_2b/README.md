# Cycle 2B review package

Cycle 2B is a published-style, single-example Online Permuted MNIST reproduction. The learner sees only `(x, y)` for each example; the evaluator uses task metadata to create and score the stream. It has no replay, task signal, model reset, or optimizer reset. Checkpoints are written at every task boundary.

## Protocol fixed before the long run

- MNIST training split; 150 tasks; every one of the 60,000 examples appears once per task.
- A fresh deterministic pixel permutation and independently seeded random example order for each task, derived from seed 101 and task index.
- `784 → 100 → 100 → 100 → 10`, ReLU hidden activations, `published_kaiming` initialization.
- SGD, learning rate 0.003, zero momentum and weight decay; batch size one.
- The per-example prediction is counted before loss/backward/update.
- 9,000,000 updates total. No tuning from the result.

The separate runner can resume from the last completed task checkpoint. Diagnostic examples are deterministic and task-permuted; diagnostics never update model or learner state. Dead means a unit returned exactly zero on every diagnostic example. Effective rank uses entropy of normalized singular values of hidden representations.

## Pre-registered descriptive gate

Compute all valid rolling 20-task means. The peak is the maximum rolling mean; final performance is the mean of tasks 130–149. `plasticity_drop = peak - final`. The gate requires a drop of at least 0.05, the peak window to end before task 130, and a sustained decline. Before inspecting the run, sustained decline is operationalized as both a negative OLS slope from the peak window's end through task 149 and a negative OLS slope over tasks 100–149. This is a descriptive pilot criterion, not an inferential test.

If any gate component fails, the run stops at Backprop and no CBP reference run is launched. If all pass, the preconfigured validated CBP reference runs with the same seed/task stream/network/optimizer and the fixed replacement settings in `configs/cycle2b/online_cbp_reference.yaml`.

## Files

- `backprop_runner_parity.json`: local online update compared with pinned official `Backprop` on copied weights and identical one-example data.
- `reproduction_task_accuracy.csv`, `reproduction_diagnostics.csv`, `reproduction_summary.json`: 150-task Backprop evidence.
- `reference_cbp_*` and `backprop_vs_cbp.csv`: created only if the Backprop gate passes.
- `sparse_initialization_sanity.csv`: small, separate real-MNIST comparison of historical dense-fan-in masking and active-fan-in Kaiming initialization; it is unrelated to the plasticity gate and is not CCDN evidence.

Full per-example model state, MNIST data, and task checkpoints remain gitignored under `results/` and `data/`.

## Recorded measurements

The prescribed Backprop run completed all 150 tasks / 9,000,000 updates on real MNIST. Peak 20-task accuracy was 0.92266 (window 0–19), final 20-task accuracy was 0.8717567, and the preregistered drop was 0.0509033. The peak was before the final region and both post-peak and final-50 task slopes were negative, so the descriptive gate passed. Across this single paired seed, CBP's peak 20-task accuracy was 0.9286508 and final 20-task accuracy was 0.9279450 (drop 0.0007058); its final-20 mean was higher than Backprop's by 0.05619. CBP ended with 0% dead units versus Backprop's 17%; mean effective rank across the three hidden layers ended at 51.15 for CBP and 25.22 for Backprop. The weighted mean absolute weight moved from 0.0601 to 0.1709 for Backprop and from 0.0601 to 0.0616 for CBP. These are descriptive pilot measurements for the reviewer; there is one seed and no significance claim.

The separate sparse initialization check recorded initial activation standard deviations 0.2824 (legacy) and 0.2611 (active-fan-in), mean adaptation AUC 0.1171 and 0.3186, and mean end accuracy 0.1288 and 0.5000, respectively. This check is a measured initialization confound, not evidence for or against CCDN.
