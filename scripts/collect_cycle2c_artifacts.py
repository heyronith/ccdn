#!/usr/bin/env python3
"""Validate and condense completed Cycle 2C results into review artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml
import torch

METHODS = ("static_sparse", "selective_reset", "rigl_reference", "ccdn_0a")
ROOT = Path(__file__).resolve().parents[1]
from scripts.run_cycle2c import file_hashes, verify_lock_file


def _rows(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def collect_run(run_root, output_dir, *, require_real=True, require_checkpoint=True):
    run_root, output_dir = Path(run_root), Path(output_dir)
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if require_real and (manifest.get("data_source") != "mnist" or manifest.get("synthetic_fallback") is not False):
        raise ValueError("Cycle 2C collector requires a real-MNIST manifest with no fallback")
    verify_lock_file(file_hashes())
    lock = json.loads((ROOT / "configs/cycle2c/protocol_lock.json").read_text())
    if manifest.get("protocol_hash") != lock.get("protocol_hash"):
        raise ValueError("run manifest protocol hash does not match the frozen protocol lock")
    if manifest.get("task_sequence_sha256") != "1f2ddb0daa90715192118466b7088fd8d3ea78f44e8647af6bca0a686393f844":
        raise ValueError("run manifest task stream hash differs from frozen protocol")
    protocol = yaml.safe_load((ROOT / "configs/cycle2c/protocol.yaml").read_text())
    protocol_path = output_dir / "protocol.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = {}
    task_rows_by_method = {}
    diagnostic_rows_by_method = {}
    for method in METHODS:
        source = run_root / method / "result"
        summary_path = source / "summary.json"
        if not summary_path.exists():
            if method != "static_sparse" and summaries.get("static_sparse") and not summaries["static_sparse"].get("loss_of_plasticity_gate_passed"):
                continue
            raise FileNotFoundError(f"completed method summary missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("tasks_completed") != 150:
            raise ValueError(f"{method} is incomplete ({summary.get('tasks_completed')}/150 tasks)")
        if summary.get("data_source") != "mnist" or summary.get("synthetic_fallback") is not False:
            raise ValueError(f"{method} summary is not verified real MNIST")
        summaries[method] = summary
        task_path, diag_path = source / "task_accuracy.csv", source / "diagnostics.csv"
        if not task_path.exists() or not diag_path.exists():
            raise FileNotFoundError(f"curve/diagnostics missing for {method}")
        task_rows_by_method[method] = _rows(task_path)
        diagnostic_rows_by_method[method] = _rows(diag_path)
        indices = [int(row["task_index"]) for row in task_rows_by_method[method]]
        if indices != list(range(150)):
            raise ValueError(f"{method} task curve is incomplete or duplicated")
        if any(not 0 <= float(row["online_accuracy"]) <= 1 for row in task_rows_by_method[method]):
            raise ValueError(f"{method} contains invalid online accuracy")
        if any(int(row.get("examples_seen_in_task", 60000)) != 60000 or
               int(row.get("optimizer_updates", 60000)) != 60000 or
               int(row.get("lifetime_examples_seen", (i + 1) * 60000)) != (i + 1) * 60000
               for i, row in enumerate(task_rows_by_method[method])):
            raise ValueError(f"{method} task update/exposure counts are invalid")
        if require_checkpoint:
            status_file = run_root / "status.json"
            if not status_file.exists() or json.loads(status_file.read_text()).get("state") not in {"COMPLETE", "STATIC_GATE_FAILED"}:
                raise ValueError("Cycle 2C run status is not a completed scientific outcome")
            checkpoint = source / "checkpoint_latest.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"final rolling checkpoint missing: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            expected = {"git_sha": manifest.get("git_commit"), "protocol_hash": manifest.get("protocol_hash"),
                        "method": method, "task_sequence_sha256": manifest.get("task_sequence_sha256")}
            if any(payload.get(key) != value for key, value in expected.items()) or payload.get("completed_task_index") != 149:
                raise ValueError(f"{method} final checkpoint identity/progress mismatch")
        shutil.copyfile(task_path, output_dir / f"{method}_task_accuracy.csv")
        shutil.copyfile(diag_path, output_dir / f"{method}_diagnostics.csv")
        (output_dir / f"{method}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not summaries:
        raise ValueError("no complete Cycle 2C methods available")

    comparison = []
    resources = []
    topology = []
    ccdn_summary = summaries.get("ccdn_0a")
    for method, summary in summaries.items():
        row = {"method": method, "resource_group": "primary_sparse", "seed": summary.get("seed", 101),
                           "final_20_task_accuracy": summary.get("final_20_task_accuracy"),
                           "plasticity_drop": summary.get("plasticity_drop"),
                           "mean_accuracy_tasks_100_149": summary.get("mean_accuracy_tasks_100_149"),
                           "final_dead_unit_fraction": summary.get("final_dead_unit_fraction"),
                           "final_mean_effective_rank": _mean(summary.get("final_effective_rank_by_layer", [])),
                           "runtime_seconds": summary.get("runtime_seconds"),
                           "estimated_total_state_memory_bytes": summary.get("resource_accounting", {}).get("estimated_total_training_state_memory_bytes")}
        if ccdn_summary and method != "ccdn_0a":
            row["ccdn_minus_final20_accuracy"] = ccdn_summary.get("final_20_task_accuracy") - summary.get("final_20_task_accuracy")
            row["ccdn_minus_plasticity_drop"] = ccdn_summary.get("plasticity_drop") - summary.get("plasticity_drop")
            row["ccdn_minus_mean_accuracy_tasks_100_149"] = ccdn_summary.get("mean_accuracy_tasks_100_149") - summary.get("mean_accuracy_tasks_100_149")
            row["ccdn_minus_final_dead_unit_fraction"] = ccdn_summary.get("final_dead_unit_fraction") - summary.get("final_dead_unit_fraction")
            row["ccdn_minus_final_mean_effective_rank"] = _mean(ccdn_summary.get("final_effective_rank_by_layer", [])) - _mean(summary.get("final_effective_rank_by_layer", []))
            row["ccdn_minus_runtime_seconds"] = ccdn_summary.get("runtime_seconds") - summary.get("runtime_seconds")
            row["ccdn_minus_estimated_total_state_memory_bytes"] = (ccdn_summary.get("resource_accounting", {}).get("estimated_total_training_state_memory_bytes") - summary.get("resource_accounting", {}).get("estimated_total_training_state_memory_bytes"))
        comparison.append(row)
        resource = dict(summary.get("resource_accounting", {}))
        resource["method"] = method
        resource["resource_group"] = "primary_sparse"
        resource["runtime_seconds"] = summary.get("runtime_seconds")
        resources.append(resource)
        diagnostics = diagnostic_rows_by_method[method]
        final_diag = diagnostics[-1]
        topology.append({"method": method,
                         "active_edges_total": final_diag.get("active_edges_total"),
                         "active_edges_layer_0": final_diag.get("active_edges_layer_0"),
                         "active_edges_layer_1": final_diag.get("active_edges_layer_1"),
                         "active_edges_layer_2": final_diag.get("active_edges_layer_2"),
                         "active_edges_layer_3": final_diag.get("active_edges_layer_3"),
                         "initial_mask_overlap_fraction_overall": final_diag.get("initial_mask_overlap_fraction_overall"),
                         "algorithm_metrics": {k: v for k, v in final_diag.items() if k.startswith("algorithm_")}})
    _write_csv(output_dir / "comparison_summary.csv", comparison)
    _write_csv(output_dir / "resource_comparison.csv", resources)
    _write_csv(output_dir / "topology_summary.csv", topology)
    _plots(task_rows_by_method, diagnostic_rows_by_method, summaries, output_dir)
    (output_dir / "README.md").write_text(
        "# Cycle 2C results\n\nPilot experiment — not definitive evidence. "
        "This package reports measurements only; interpretation is reserved for the reviewer.\n\n"
        f"Run ID: `{manifest.get('run_id')}`. Methods included: {', '.join(summaries)}.\n")
    return {"methods": list(summaries), "output_dir": str(output_dir)}


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plots(task_rows, diagnostic_rows, summaries, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, rows in task_rows.items():
        ax.plot([int(r["task_index"]) for r in rows], [float(r["online_accuracy"]) for r in rows], label=method)
    ax.set(xlabel="Task", ylabel="Online accuracy (pre-update)", title="Cycle 2C online accuracy")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "online_accuracy_comparison.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(summaries)
    ax.bar(methods, [summaries[m].get("plasticity_drop", 0) for m in methods])
    ax.set(ylabel="Peak minus final-20 online accuracy", title="Plasticity drop (descriptive)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output_dir / "plasticity_drop_comparison.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, rows in diagnostic_rows.items():
        tasks, edges = [], []
        for row in rows:
            key = "algorithm_edges_pruned"
            if key in row and row[key] not in (None, ""):
                tasks.append(int(row["task_index"]))
                edges.append(int(float(row[key])))
            elif row.get("algorithm_reset_count") not in (None, ""):
                tasks.append(int(row["task_index"]))
                edges.append(int(float(row["algorithm_reset_count"])))
        if tasks: ax.plot(tasks, edges, marker=".", label=method)
    ax.set(xlabel="Task", ylabel="Cumulative edges pruned / weights reset", title="Structural intervention counters")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "topology_turnover.png", dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default="review_artifacts/cycle_2c")
    args = parser.parse_args()
    print(json.dumps(collect_run(args.run_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
