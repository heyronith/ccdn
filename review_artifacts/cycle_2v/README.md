# Cycle 2V review package

**Purpose:** acquire pinned official Continual Backprop and RigL implementations, execute compact real-MNIST smokes, and validate local reference ports against upstream behavior. This package contains no CCDN performance comparison and makes no scientific performance claim.

## Pinned sources

- Continual Backprop: `shibhansh/loss-of-plasticity`, `a6b79580d85f3025bdb601566d3627c5f489f13b`.
- RigL: `google-research/rigl`, `d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9`.

The exact upstream files are fetched to gitignored `.external/official_baselines/`. Neither source checkout is modified or committed. `provenance.json` records verified remotes/SHAs and the Cycle 2 base commit. The local reference ports live in `ccdn/official_reference/`; public wrappers are in `ccdn/baselines/*_reference.py`.

## Reproduce validation

Fetch the pinned sources with `python scripts/fetch_official_baselines.py`. Install each project's dependencies in its own Python 3.11 environment (see `environment_versions.json`; CBP uses PyTorch 2.1.0 / torchvision 0.16.0, RigL uses TensorFlow 2.15.1 and TFDS 4.9.4). Then run:

```bash
python -m ccdn.validation.official_baselines
```

That command checks the pins, compares local ports with direct upstream traces, and refreshes compact evidence. Add `--full` to rerun both official-code real-MNIST execution smokes. RigL uses an external runtime wrapper to select legacy Keras SGD APIs under TensorFlow 2.15; the pinned algorithm source remains pristine. Its upstream trainer also uses TFDS.

## Evidence and limits

- `official_cbp_execution.json`: pinned PyTorch upstream loader and online experiment ran on real MNIST; six large-batch updates across two 30,000-example task segments. The low accuracy in this intentionally tiny execution smoke is not a performance result.
- `official_rigl_execution.json`: pinned TensorFlow upstream trainer ran three optimizer updates on MNIST. The initial 80% one-shot pruning allows RigL to exercise actual mask updates; per-layer active edge counts were conserved while all three masks changed.
- `cbp_parity.json` and `rigl_parity.json`: implementation parity probes and structural checks.
- `official_cbp_mechanism.json`: a separate CBP mechanism probe for utility, maturity, fractional replacement, initialization, and reset state.
- `environment_versions.json`, `provenance.json`, `validation_summary.json`: runtime, source, and compact status records.
- `legacy_vs_reference.md`: behavioral differences from the Cycle 2 lightweight methods.

These checks validate acquisition, execution, and selected algorithm mechanics. The probes do not establish end-to-end scientific equivalence across all configurations or benchmark performance. No CCDN comparison was run.
