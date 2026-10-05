# Cycle 2 approximations vs Cycle 2V references

This note distinguishes the historical Cycle 2 local baselines from the separately named Cycle 2V ports. The ports were compared with focused traces from the pinned sources. This is not a claim of full equivalence for every upstream option or training setup.

## Continual Backprop

The Cycle 2 lightweight `continual_backprop` uses a simpler activation-times-sensitivity utility and cadence-based hidden-unit replacement. `continual_backprop_reference` follows pinned upstream Generate-and-Test behavior: adaptable-contribution utility, bias-corrected utility, age/maturity eligibility, and a fractional replacement accumulator. It also reproduces upstream initialization bounds and outgoing-weight reset behavior.

The upstream reset sequence clears a unit's activation average before the following outgoing-bias correction reads that average. The reference port preserves that observable ordering because parity is the purpose of this port, even though it is surprising. The direct trace compared weights/biases, utility, corrected utility, age, accumulator, and reset outcomes over four updates, including seven replacements. The official upstream execution smoke is separate and used two 30,000-example segments with batches of 10,000; its metrics are only execution evidence.

## RigL

The Cycle 2 local `rigl` is a lightweight implementation with its own cadence/schedule decisions. `rigl_reference` ports the pinned `SparseRigLOptimizerBase` update policy: prune by active magnitude, select growth candidates by gradient score, and maintain the edge budget. Newly grown, previously inactive connections are zero initialized and optimizer state is cleared. Following upstream behavior, an edge pruned and immediately selected again by the growth ranking may retain its stored value and optimizer slot.

Candidate gradients are provided by the local practical dense candidate-gradient machinery. The local implementation's schedule and minimal experiment interface do not reproduce the full TensorFlow trainer. A direct one-step non-tie trace against upstream matched masks and weights and checked pruning, gradient-ranked growth, exact edge conservation, zero initialization, and optimizer-state reset. The official MNIST smoke additionally confirmed all three layer masks changed while active counts stayed `[5018, 205, 64]`.

## Usage guidance

Use the `*_reference` names when an experiment intends to use these validated reference semantics. Existing Cycle 2 result files and local approximations remain unchanged; Cycle 2V did not rerun or compare their benchmark performance.
