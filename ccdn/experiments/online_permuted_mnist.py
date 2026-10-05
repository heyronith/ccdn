"""Published-style, one-example-at-a-time Online Permuted MNIST runner.

The evaluator creates task permutations/orders and logs task indices. The
learner interface is strictly ``(x, y)`` plus ordinary optimization hooks; no
task ID or boundary callback is sent to it. Checkpoints are written after
every completed task and contain the optimizer, algorithm, RNG, and progress.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ccdn.baselines import StaticDense
from ccdn.official_reference import ContinualBackpropReference
from ccdn.models.dense_mlp import DenseMLP
from ccdn.metrics.plasticity import online_plasticity_diagnostics
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from ccdn.utils.config import load_config, save_config
from ccdn.utils.reproducibility import (capture_rng_state, restore_rng_state,
                                       seed_everything)


def _device(requested):
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def _learner_step(model, algorithm, optimizer, x, y):
    """One predict-before-update online example; receives no task metadata."""
    optimizer.zero_grad(set_to_none=False)
    logits = model(x)
    prediction = int(logits.argmax(dim=1).item())
    loss = F.cross_entropy(logits, y)
    algorithm.before_backward(loss)
    loss.backward()
    algorithm.after_backward()
    optimizer.step()
    algorithm.after_optimizer_step(optimizer)
    return prediction, float(loss.detach())


def _slope(values, start, end):
    if start < 0 or end >= len(values) or end <= start:
        return None
    x = np.arange(start, end + 1, dtype=np.float64)
    y = np.asarray(values[start:end + 1], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def summarize_online(task_rows, diagnostic_rows, requested_tasks,
                     runtime_seconds, resumed):
    accuracy = [float(row["online_accuracy"]) for row in task_rows]
    completed = len(accuracy)
    summary = {
        "protocol": "single-example online permuted MNIST; prediction before update",
        "tasks_requested": int(requested_tasks),
        "tasks_completed": completed,
        "total_optimizer_updates": int(sum(int(r["optimizer_updates"]) for r in task_rows)),
        "runtime_seconds": float(runtime_seconds),
        "resumed_from_checkpoint": bool(resumed),
        "peak_20_task_accuracy": None,
        "peak_task": None,
        "peak_window_start": None,
        "peak_window_end": None,
        "final_20_task_accuracy": None,
        "plasticity_drop": None,
        "loss_of_plasticity_gate_passed": None,
        "accuracy_task_0": accuracy[0] if accuracy else None,
        "accuracy_task_149": accuracy[149] if completed > 149 else None,
        "slope_tasks_0_49": _slope(accuracy, 0, 49),
        "slope_tasks_50_99": _slope(accuracy, 50, 99),
        "slope_tasks_100_149": _slope(accuracy, 100, 149),
    }
    if completed >= 20:
        rolling = np.convolve(np.asarray(accuracy), np.ones(20) / 20,
                              mode="valid")
        peak_start = int(np.argmax(rolling))
        peak = float(rolling[peak_start])
        final = float(np.mean(accuracy[-20:])) if completed >= requested_tasks else None
        drop = peak - final if final is not None else None
        peak_end = peak_start + 19
        peak_before_final = peak_end < requested_tasks - 20
        slope_after_peak = _slope(accuracy, peak_end, completed - 1)
        late_slope = _slope(accuracy, max(0, completed - 50), completed - 1)
        sustained = (slope_after_peak is not None and late_slope is not None
                     and slope_after_peak < 0 and late_slope < 0)
        passed = (completed == requested_tasks and final is not None
                  and drop >= 0.05 and peak_before_final and sustained)
        summary.update({
            "peak_20_task_accuracy": peak,
            "peak_task": peak_end,
            "peak_window_start": peak_start,
            "peak_window_end": peak_end,
            "final_20_task_accuracy": final,
            "plasticity_drop": drop,
            "peak_occurs_before_final_20_task_region": peak_before_final,
            "slope_from_peak_window_end_to_final": slope_after_peak,
            "sustained_downward_trend": sustained,
            "loss_of_plasticity_gate_passed": bool(passed) if completed == requested_tasks else None,
        })
    if diagnostic_rows:
        first = diagnostic_rows[0]
        last = diagnostic_rows[-1]
        summary.update({
            "initial_dead_unit_fraction": first.get("dead_unit_fraction_overall"),
            "final_dead_unit_fraction": last.get("dead_unit_fraction_overall"),
            "initial_mean_absolute_weight": first.get("mean_absolute_weight"),
            "final_mean_absolute_weight": last.get("mean_absolute_weight"),
            "initial_effective_rank_by_layer": [first.get(f"effective_rank_layer_{i}") for i in range(3)],
            "final_effective_rank_by_layer": [last.get(f"effective_rank_layer_{i}") for i in range(3)],
        })
    return summary


def write_plots(task_rows, diagnostic_rows, output_dir, overlay=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["task_index"] for r in task_rows],
            [r["online_accuracy"] for r in task_rows], label="Backprop")
    if overlay:
        ax.plot([r["task_index"] for r in overlay],
                [r["online_accuracy"] for r in overlay], label="CBP Reference")
    ax.set(xlabel="Task", ylabel="Online accuracy (pre-update)",
           title="Online Permuted MNIST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "online_accuracy_by_task.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i in range(3):
        axes[0].plot([r["task_index"] for r in diagnostic_rows],
                     [r.get(f"dead_unit_fraction_layer_{i}") for r in diagnostic_rows],
                     label=f"Layer {i}")
        axes[1].plot([r["task_index"] for r in diagnostic_rows],
                     [r.get(f"mean_absolute_weight_layer_{i}") for r in diagnostic_rows],
                     label=f"Layer {i}")
        axes[2].plot([r["task_index"] for r in diagnostic_rows],
                     [r.get(f"effective_rank_layer_{i}") for r in diagnostic_rows],
                     label=f"Layer {i}")
    axes[0].set_ylabel("Dead-unit fraction")
    axes[1].set_ylabel("Mean |weight|")
    axes[2].set_ylabel("Effective rank")
    axes[2].set_xlabel("Diagnostic task index")
    for ax in axes:
        ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "plasticity_diagnostics.png", dpi=140)
    plt.close(fig)
    return True


def _write_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path, integer_fields=()):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = None
                elif key in integer_fields:
                    converted[key] = int(value)
                elif key == "phase":
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def _save_task_checkpoint(path, model, algorithm, optimizer, task_index,
                          lifetime_examples, runtime_seconds, config):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "algorithm": algorithm.state_dict(),
        "completed_task_index": int(task_index),
        "lifetime_examples_seen": int(lifetime_examples),
        "rng_state": capture_rng_state(),
        "config": copy.deepcopy(config),
        "runtime_seconds": float(runtime_seconds),
    }, temp)
    temp.replace(path)


def run_online_experiment(config, *, stream=None, output_dir=None,
                          resume_from=None, max_tasks_this_invocation=None):
    started = time.perf_counter()
    cfg = copy.deepcopy(config)
    exp, model_cfg, opt_cfg = cfg["experiment"], cfg["model"], cfg["optimizer"]
    seed = int(exp.get("seed", 101))
    seed_everything(seed, exp.get("deterministic", True))
    device = _device(exp.get("device", "auto"))
    if device.type == "cpu":
        torch.set_num_threads(int(exp.get("cpu_threads", 1)))
    stream = stream or OnlinePermutedMNIST(
        seed=seed, root=cfg.get("stream", {}).get("dataset_root", "./data"))
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [100, 100, 100]))
    if len(hidden_sizes) != 3:
        raise ValueError("published-style reproduction requires exactly three hidden layers")
    model = DenseMLP(input_size=stream.input_size, hidden_sizes=hidden_sizes,
                     output_size=int(model_cfg.get("output_size", 10)),
                     initialization=model_cfg.get("initialization", "published_kaiming")).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(opt_cfg.get("learning_rate", 0.003)),
                                momentum=float(opt_cfg.get("momentum", 0.0)),
                                weight_decay=float(opt_cfg.get("weight_decay", 0.0)))
    algorithm_name = model_cfg.get("type", "static_dense")
    if algorithm_name == "static_dense":
        algorithm = StaticDense(model)
    elif algorithm_name == "continual_backprop_reference":
        algorithm = ContinualBackpropReference(model, **cfg.get("algorithm", {}))
    else:
        raise ValueError(f"unsupported online learner: {algorithm_name}")

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        algorithm.load_state_dict(checkpoint["algorithm"])
        restore_rng_state(checkpoint["rng_state"])
        first_task = int(checkpoint["completed_task_index"]) + 1
        lifetime_examples = int(checkpoint["lifetime_examples_seen"])
        runtime_before = float(checkpoint.get("runtime_seconds", 0.0))
        saved_config = checkpoint["config"]
        if saved_config != cfg:
            raise ValueError("resume configuration differs from checkpoint configuration")
        output_dir = Path(output_dir or Path(resume_from).parent)
        resumed = True
    else:
        first_task = 0
        lifetime_examples = 0
        runtime_before = 0.0
        resumed = False
        if output_dir is None:
            root = Path(cfg.get("output", {}).get("root", "results"))
            label = exp.get("name", "cycle_2b")
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            output_dir = root / label / algorithm_name / f"seed_{seed}" / f"{stamp}_{uuid.uuid4().hex[:8]}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        save_config(cfg, output_dir / "config.yaml")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_csv = output_dir / "task_accuracy.csv"
    diagnostic_csv = output_dir / "diagnostics.csv"
    task_rows = _read_rows(task_csv, {"task_index", "correct_predictions",
                                     "examples_seen_in_task", "lifetime_examples_seen",
                                     "optimizer_updates"})
    diagnostic_rows = _read_rows(diagnostic_csv, {"task_index", "lifetime_examples_seen"})

    tasks_requested = int(cfg.get("stream", {}).get("tasks", 150))
    examples_per_task = int(cfg.get("stream", {}).get("examples_per_task", 60000))
    if examples_per_task > len(stream.labels):
        raise ValueError("examples_per_task exceeds available MNIST training examples")
    diagnostic_examples = int(cfg.get("diagnostics", {}).get("examples", 2000))
    task_limit = tasks_requested
    if max_tasks_this_invocation is not None:
        task_limit = min(task_limit, first_task + int(max_tasks_this_invocation))

    for task_index in range(first_task, task_limit):
        task = stream.task(task_index)
        if task_index == 0 or task_index % 5 == 0:
            diag_x, _ = stream.diagnostic_batch(task, diagnostic_examples, device)
            diagnostic_rows.append({"task_index": task_index, "phase": "task_start",
                                    "lifetime_examples_seen": lifetime_examples,
                                    **online_plasticity_diagnostics(model, diag_x)})
            _write_rows(diagnostic_rows, diagnostic_csv)

        correct = 0
        loss_sum = 0.0
        for position in range(examples_per_task):
            x, y = stream.sample(task, position, device)
            # Learner-facing API contains only x and y.
            prediction, loss = _learner_step(model, algorithm, optimizer, x, y)
            correct += int(prediction == int(y.item()))
            loss_sum += loss
            lifetime_examples += 1
        task_rows.append({
            "task_index": task_index,
            "online_accuracy": correct / examples_per_task,
            "correct_predictions": correct,
            "examples_seen_in_task": examples_per_task,
            "lifetime_examples_seen": lifetime_examples,
            "optimizer_updates": examples_per_task,
            "mean_preupdate_loss": loss_sum / examples_per_task,
        })
        _write_rows(task_rows, task_csv)
        cumulative_runtime = runtime_before + time.perf_counter() - started
        checkpoint_path = output_dir / f"checkpoint_task_{task_index + 1:03d}.pt"
        _save_task_checkpoint(checkpoint_path, model, algorithm, optimizer,
                              task_index, lifetime_examples, cumulative_runtime, cfg)
        print(f"completed task {task_index + 1}/{tasks_requested}: "
              f"online_accuracy={correct / examples_per_task:.6f}, "
              f"updates={lifetime_examples}, checkpoint={checkpoint_path}",
              flush=True)

    completed = len(task_rows)
    complete_run = completed == tasks_requested
    if complete_run and tasks_requested:
        final_index = tasks_requested - 1
        has_final = any(row.get("phase") == "final" and
                        int(row.get("task_index", -1)) == final_index
                        for row in diagnostic_rows)
        if not has_final:
            final_task = stream.task(final_index)
            diag_x, _ = stream.diagnostic_batch(final_task, diagnostic_examples, device)
            diagnostic_rows.append({"task_index": final_index, "phase": "final",
                                    "lifetime_examples_seen": lifetime_examples,
                                    **online_plasticity_diagnostics(model, diag_x)})
            _write_rows(diagnostic_rows, diagnostic_csv)
    runtime_total = runtime_before + time.perf_counter() - started
    summary = {
        "protocol": "single-example online permuted MNIST",
        "learner_type": algorithm_name,
        "seed": seed,
        "tasks_requested": tasks_requested,
        "tasks_completed": completed,
        "examples_per_task": examples_per_task,
        "total_optimizer_updates": int(sum(int(r["optimizer_updates"]) for r in task_rows)),
        "architecture": [stream.input_size, *hidden_sizes, int(model_cfg.get("output_size", 10))],
        "activation": "relu",
        "initialization": model_cfg.get("initialization", "published_kaiming"),
        "learning_rate": float(opt_cfg.get("learning_rate", 0.003)),
        "momentum": float(opt_cfg.get("momentum", 0.0)),
        "weight_decay": float(opt_cfg.get("weight_decay", 0.0)),
        "stream_seed": seed,
        "data_source": stream.data_source,
        "synthetic_fallback": False,
        "stream_metadata": stream.metadata,
        "task_sequence_sha256": stream.sequence_digest(tasks_requested, examples_per_task),
        "runtime_seconds": runtime_total,
        "resumed_from_checkpoint": resumed,
        "final_replacement_state": algorithm.metrics(),
    }
    if isinstance(algorithm, ContinualBackpropReference):
        summary["replacement_count"] = int(algorithm.replacement_count)
        summary["replacement_rate"] = algorithm.replacement_rate
        summary["final_replacement_state"] = {
            **algorithm.metrics(),
            "replacement_rate": algorithm.replacement_rate,
            "decay_rate": algorithm.decay_rate,
            "maturity_threshold": algorithm.maturity_threshold,
            "accumulate": algorithm.accumulate,
            "util_type": algorithm.util_type,
            "accumulated_replacements_by_layer": list(algorithm.accumulated_replacements),
            "eligible_units_by_layer": [int((age > algorithm.maturity_threshold).sum())
                                        for age in algorithm.age],
            "mean_age_by_layer": [float(age.float().mean()) for age in algorithm.age],
            "mean_utility_by_layer": [float(value.mean()) for value in algorithm.utility],
            "mean_bias_corrected_utility_by_layer": [
                float(value.mean()) for value in algorithm.bias_corrected_utility],
        }
    summary.update(summarize_online(task_rows, diagnostic_rows, tasks_requested,
                                    runtime_total, resumed))
    if not complete_run:
        summary["loss_of_plasticity_gate_passed"] = None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if complete_run:
        write_plots(task_rows, diagnostic_rows, output_dir)
    print(output_dir)
    return output_dir, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-tasks-this-invocation", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    run_online_experiment(config, resume_from=args.resume_from,
                          max_tasks_this_invocation=args.max_tasks_this_invocation)


if __name__ == "__main__":
    main()
