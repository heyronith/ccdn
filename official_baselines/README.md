# Official baseline provenance

The source repositories are fetched by `scripts/fetch_official_baselines.py` into gitignored `.external/official_baselines/` and detached at the immutable SHAs in `provenance.yaml`. The fetch script verifies every resulting `HEAD` and rejects origin mismatches.

CCDN's existing Cycle 2 methods remain historical local approximations (`continual_backprop` and `rigl` aliases). They are not official implementations. Cycle 2V adds separately named reference ports. Their status and evidence are recorded in `review_artifacts/cycle_2v/validation_summary.json`.

Run `python scripts/fetch_official_baselines.py` to fetch and verify the source trees. Environment setup is intentionally kept separate for the PyTorch CBP and TensorFlow RigL upstream projects; see the committed environment versions and runtime compatibility note in the review package. Run `python -m ccdn.validation.official_baselines` to repeat tensor-level parity validation against local fetched clones and refresh the evidence. `--full` also reruns the small official MNIST execution smokes.
