"""Build the compact, paired Cycle 2B review package from completed runs."""
import argparse
import csv
import json
from pathlib import Path

import torch

from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "review_artifacts/cycle_2b"


def _single_summary(search_root):
    paths = list(search_root.glob("**/summary.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one completed summary under {search_root}, found {len(paths)}")
    return paths[0]


def _rows(path):
    with open(path, newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _correct_global_weight_mean(rows, architecture):
    layer_sizes = [architecture[i] * architecture[i + 1]
                   for i in range(len(architecture) - 1)]
    names = [f"mean_absolute_weight_layer_{i}" for i in range(len(architecture) - 2)]
    names.append("mean_absolute_weight_output")
    denominator = sum(layer_sizes)
    for row in rows:
        row["mean_absolute_weight"] = sum(
            float(row[name]) * count
            for name, count in zip(names, layer_sizes)) / denominator
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp-run", type=Path)
    parser.add_argument("--cbp-run", type=Path)
    args = parser.parse_args()
    bp_summary_path = args.bp_run or _single_summary(ROOT / "results/cycle_2b_online_bp")
    cbp_summary_path = args.cbp_run or _single_summary(ROOT / "results/cycle_2b_online_cbp")
    bp_dir, cbp_dir = bp_summary_path.parent, cbp_summary_path.parent
    bp = json.loads(bp_summary_path.read_text())
    cbp = json.loads(cbp_summary_path.read_text())

    # Hash the deterministic task permutations and orders for both runs. The
    # MNIST tensor is loaded once per stream; only its sequence digest is saved.
    stream_bp = OnlinePermutedMNIST(seed=bp["seed"], root=ROOT / "data", download=False)
    stream_cbp = OnlinePermutedMNIST(seed=cbp["seed"], root=ROOT / "data", download=False)
    bp["data_source"] = stream_bp.data_source
    cbp["data_source"] = stream_cbp.data_source
    bp["synthetic_fallback"] = cbp["synthetic_fallback"] = False
    bp["peak_task"] = bp["peak_window_end"]
    cbp["peak_task"] = cbp["peak_window_end"]
    bp["task_sequence_sha256"] = stream_bp.sequence_digest(bp["tasks_requested"], bp["examples_per_task"])
    cbp["task_sequence_sha256"] = stream_cbp.sequence_digest(cbp["tasks_requested"], cbp["examples_per_task"])
    pairing_verified = (bp["seed"] == cbp["seed"]
                       and bp["data_source"] == cbp["data_source"] == "mnist"
                       and bp["stream_metadata"] == cbp["stream_metadata"]
                       and bp["task_sequence_sha256"] == cbp["task_sequence_sha256"]
                       and bp["architecture"] == cbp["architecture"]
                       and bp["initialization"] == cbp["initialization"]
                       and bp["learning_rate"] == cbp["learning_rate"]
                       and bp["examples_per_task"] == cbp["examples_per_task"]
                       and bp["tasks_requested"] == cbp["tasks_requested"])
    if not pairing_verified:
        raise RuntimeError("Backprop and CBP do not share the same protocol/task sequence")

    cbp["official_reference_source"] = "shibhansh/loss-of-plasticity"
    cbp["official_reference_commit"] = "a6b79580d85f3025bdb601566d3627c5f489f13b"
    cbp["replacement_rate"] = 1e-5
    cbp_state = torch.load(cbp_dir / "checkpoint_task_150.pt",
                           map_location="cpu", weights_only=False)["algorithm"]
    cbp["replacement_count"] = int(cbp_state["replacement_count"])
    cbp["final_replacement_state"] = {
        "cbp_reference_replacements": int(cbp_state["replacement_count"]),
        "replacement_rate": 1e-5,
        "decay_rate": 0.99,
        "maturity_threshold": 100,
        "accumulate": True,
        "util_type": "adaptable_contribution",
        "accumulated_replacements_by_layer": cbp_state["accumulated_replacements"],
        "eligible_units_by_layer": [int((age > 100).sum()) for age in cbp_state["age"]],
        "mean_age_by_layer": [float(age.float().mean()) for age in cbp_state["age"]],
        "mean_utility_by_layer": [float(value.mean()) for value in cbp_state["utility"]],
        "mean_bias_corrected_utility_by_layer": [
            float(value.mean()) for value in cbp_state["bias_corrected_utility"]],
    }
    cbp["loss_of_plasticity_gate_passed"] = None

    ART.mkdir(parents=True, exist_ok=True)
    bp_task = _rows(bp_dir / "task_accuracy.csv")
    cbp_task = _rows(cbp_dir / "task_accuracy.csv")
    if len(bp_task) != 150 or len(cbp_task) != 150:
        raise RuntimeError("both primary runs must contain all 150 completed tasks")
    bp_diag = _correct_global_weight_mean(_rows(bp_dir / "diagnostics.csv"), bp["architecture"])
    cbp_diag = _correct_global_weight_mean(_rows(cbp_dir / "diagnostics.csv"), cbp["architecture"])
    for rows, summary in ((bp_diag, bp), (cbp_diag, cbp)):
        initial = next(r for r in rows if r["phase"] == "task_start")
        final = next(r for r in rows if r["phase"] == "final")
        summary["initial_mean_absolute_weight"] = float(initial["mean_absolute_weight"])
        summary["final_mean_absolute_weight"] = float(final["mean_absolute_weight"])
    bp["paired_task_sequence_verified"] = pairing_verified
    cbp["paired_task_sequence_verified"] = pairing_verified
    bp_summary_path.write_text(json.dumps(bp, indent=2) + "\n")
    cbp_summary_path.write_text(json.dumps(cbp, indent=2) + "\n")

    bp_task_path, cbp_task_path = bp_dir / "task_accuracy.csv", cbp_dir / "task_accuracy.csv"
    _write(ART / "reproduction_task_accuracy.csv", _rows(bp_task_path))
    _write(ART / "reproduction_diagnostics.csv", bp_diag)
    _write(ART / "reference_cbp_task_accuracy.csv", _rows(cbp_task_path))
    _write(ART / "reference_cbp_diagnostics.csv", cbp_diag)
    _write(bp_dir / "diagnostics.csv", bp_diag)
    _write(cbp_dir / "diagnostics.csv", cbp_diag)
    (ART / "reproduction_summary.json").write_text(json.dumps(bp, indent=2) + "\n")
    (ART / "reference_cbp_summary.json").write_text(json.dumps(cbp, indent=2) + "\n")

    comparison = []
    for i, (b, c) in enumerate(zip(bp_task, cbp_task)):
        bacc, cacc = float(b["online_accuracy"]), float(c["online_accuracy"])
        bstart, cstart = max(0, i - 19), max(0, i - 19)
        broll = sum(float(r["online_accuracy"]) for r in bp_task[bstart:i + 1]) / (i - bstart + 1)
        croll = sum(float(r["online_accuracy"]) for r in cbp_task[cstart:i + 1]) / (i - cstart + 1)
        comparison.append({
            "task_index": i,
            "backprop_online_accuracy": bacc,
            "cbp_reference_online_accuracy": cacc,
            "cbp_minus_backprop": cacc - bacc,
            "backprop_rolling_20_task_accuracy": broll if i >= 19 else "",
            "cbp_rolling_20_task_accuracy": croll if i >= 19 else "",
        })
    _write(ART / "backprop_vs_cbp.csv", comparison)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; plots were not generated")
    else:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(150), [float(r["online_accuracy"]) for r in bp_task], label="Backprop")
        ax.plot(range(150), [float(r["online_accuracy"]) for r in cbp_task], label="CBP Reference")
        ax.set(xlabel="Task", ylabel="Online accuracy (pre-update)",
               title="Online Permuted MNIST — pilot reproduction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ART / "online_accuracy_by_task.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for label, rows, color in (("Backprop", bp_diag, "tab:blue"),
                                   ("CBP Reference", cbp_diag, "tab:orange")):
            x = [int(r["task_index"]) for r in rows]
            axes[0].plot(x, [float(r["dead_unit_fraction_overall"]) for r in rows],
                         color=color, label=label)
            axes[1].plot(x, [float(r["mean_absolute_weight"]) for r in rows],
                         color=color, label=label)
            rank = [sum(float(r[f"effective_rank_layer_{i}"]) for i in range(3)) / 3
                    for r in rows]
            axes[2].plot(x, rank, color=color, label=label)
        axes[0].set_ylabel("Dead-unit fraction")
        axes[1].set_ylabel("Mean |weight|")
        axes[2].set_ylabel("Mean effective rank")
        axes[2].set_xlabel("Diagnostic task index")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(ART / "plasticity_diagnostics.png", dpi=150)
        plt.close(fig)
    print(f"wrote Cycle 2B artifacts to {ART}")


if __name__ == "__main__":
    main()
