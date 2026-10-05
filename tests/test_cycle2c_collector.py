import csv
import json
from pathlib import Path

from scripts.collect_cycle2c_artifacts import collect_run


def test_collector_uses_fixture_data_and_produces_compact_tables(tmp_path):
    run = tmp_path / "fixture"
    run.mkdir()
    lock = json.loads(Path("configs/cycle2c/protocol_lock.json").read_text())
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "fixture", "data_source": "mnist",
                                                        "synthetic_fallback": False,
                                                        "protocol_hash": lock["protocol_hash"],
                                                        "task_sequence_sha256": "1f2ddb0daa90715192118466b7088fd8d3ea78f44e8647af6bca0a686393f844"}))
    method = "static_sparse"
    result = run / method / "result"
    result.mkdir(parents=True)
    (result / "summary.json").write_text(json.dumps({
        "tasks_completed": 150, "seed": 101, "final_20_task_accuracy": .5,
        "data_source": "mnist", "synthetic_fallback": False,
        "plasticity_drop": .1, "mean_accuracy_tasks_100_149": .4,
        "final_dead_unit_fraction": .2, "final_effective_rank_by_layer": [2, 2, 2],
        "runtime_seconds": 42,
        "resource_accounting": {"estimated_total_training_state_memory_bytes": 1234},
    }))
    with open(result / "task_accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_index", "online_accuracy"])
        writer.writeheader()
        writer.writerows({"task_index": i, "online_accuracy": .5} for i in range(150))
    with open(result / "diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_index", "active_edges_total"])
        writer.writeheader()
        writer.writerow({"task_index": 149, "active_edges_total": 98515})
    output = tmp_path / "review"
    result_info = collect_run(run, output, require_real=True, require_checkpoint=False)
    assert result_info["methods"] == [method]
    assert (output / "static_sparse_task_accuracy.csv").exists()
    assert (output / "resource_comparison.csv").exists()
    assert (output / "topology_summary.csv").exists()
