# Cycle 1 smoke review artifacts

> Synthetic integration smoke test only. These are not scientific MNIST results.

Contains the fresh synthetic smoke matrix rerun after the Cycle 1 review fixes: five baselines × seeds 1 and 2, with 20 permutations per run. Every run directory preserves its resolved config, `summary.json`, `adaptation_auc.csv`, and full `metrics.csv`.

The summaries and per-evaluation rows are also collected in `smoke_summaries.csv` and `smoke_metrics.csv`. `manifest.json` lists each source run and copied artifact. These outputs are integration evidence only.
