"""Measure the Cycle 2 sparse initialization confound on a tiny real-MNIST run."""
import csv
import json
from pathlib import Path

from ccdn.experiments.train import run
from ccdn.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _first_activation_std(path):
    with open(path, newline="", encoding="utf8") as handle:
        row = next(csv.DictReader(handle))
    return float(row["activation_std"])


def main():
    base = load_config(ROOT / "configs/suites/cycle2_preflight.yaml")["base"]
    records = []
    for mode in ("legacy", "active_fan_in_kaiming"):
        config = json.loads(json.dumps(base))
        config.setdefault("experiment", {}).update({
            "name": f"cycle2b_sparse_init_{mode}", "seed": 11,
        })
        config["stream"].update({
            "permutations": 5,
            "batches_per_permutation": 20,
            "batch_size": 128,
            "synthetic": False,
            "allow_synthetic_fallback": False,
        })
        config["model"].update({"type": "static_sparse", "initialization": mode})
        run_dir = run(config)
        summary = json.loads((run_dir / "summary.json").read_text())
        auc_rows = list(csv.DictReader((run_dir / "adaptation_auc.csv").open()))
        records.append({
            "initialization": mode,
            "seed": 11,
            "data_source": summary["data_source"],
            "synthetic_fallback": summary["synthetic_fallback"],
            "initial_activation_std": _first_activation_std(run_dir / "metrics.csv"),
            "early_adaptation_auc": sum(float(row["adaptation_auc"]) for row in auc_rows) / len(auc_rows),
            "mean_end_accuracy": summary["primary_metrics"]["mean_end_accuracy"],
            "learning_rate": config["optimizer"]["lr"],
            "density": config["model"]["density"],
            "hidden_sizes": config["model"]["hidden_sizes"],
            "run_dir": str(run_dir),
        })
    output = ROOT / "review_artifacts/cycle_2b/sparse_initialization_sanity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(output)


if __name__ == "__main__":
    main()
