import json

import matplotlib.pyplot as plt
import numpy as np

from src.workloads import (
    generate_sequential_workload,
    generate_random_workload,
    generate_bursty_workload,
    generate_shifting_trace,
)
from src.simulate import run_shifting_experiment


def _summarize_confidence(history):
    correct_rows = [row for row in history if row["correct"]]
    incorrect_rows = [row for row in history if not row["correct"]]

    mean_confidence_correct = (
        float(np.mean([row["confidence"] for row in correct_rows]))
        if correct_rows
        else None
    )
    mean_confidence_incorrect = (
        float(np.mean([row["confidence"] for row in incorrect_rows]))
        if incorrect_rows
        else None
    )

    calibration_bins = []
    for index in range(5):
        lower = index / 5
        upper = (index + 1) / 5
        rows = [
            row
            for row in history
            if lower <= row["confidence"]
            and (row["confidence"] < upper or (index == 4 and row["confidence"] <= upper))
        ]
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(rows),
                "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
                "mean_confidence": float(np.mean([row["confidence"] for row in rows]))
                if rows
                else None,
            }
        )

    return {
        "window_count": len(history),
        "accuracy": float(np.mean([row["correct"] for row in history])) if history else None,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "confidence_gap": (
            mean_confidence_correct - mean_confidence_incorrect
            if mean_confidence_correct is not None and mean_confidence_incorrect is not None
            else None
        ),
        "brier_score": float(
            np.mean([(row["confidence"] - float(row["correct"])) ** 2 for row in history])
        )
        if history
        else None,
        "calibration_bins": calibration_bins,
    }


def _summarize_shift_phases(history, shift_step):
    summaries = {}
    for phase, predicate in (
        ("before_shift", lambda row: row["step"] < shift_step),
        ("after_shift", lambda row: row["step"] >= shift_step),
    ):
        rows = [row for row in history if predicate(row)]
        summaries[phase] = {
            "window_count": len(rows),
            "static_seeks": {
                algorithm: sum(row["static_costs"][algorithm] for row in rows)
                for algorithm in ("FCFS", "SSTF", "SCAN", "CSCAN")
            },
            "learned_seek": sum(row["seek_cost"] for row in rows),
            "selector_accuracy": float(np.mean([row["correct"] for row in rows]))
            if rows
            else None,
        }
    return summaries


def main():
    traces = {
        "sequential": generate_sequential_workload(length=200),
        "random": generate_random_workload(length=200, seed=17),
        "bursty": generate_bursty_workload(length=200, seed=17),
        "shifting": generate_shifting_trace(total_length=200, seed=42),
    }

    scenarios = {}
    for name, trace in traces.items():
        np.random.seed(42)
        static_results, learned_result, history = run_shifting_experiment(
            trace, window_size=10
        )
        scenario = {
            "static_seeks": static_results,
            "learned_seek": learned_result,
            "history": history,
            "confidence_analysis": _summarize_confidence(history),
        }
        if name == "shifting":
            scenario["phase_seeks"] = _summarize_shift_phases(
                history, len(trace) // 2
            )
        scenarios[name] = scenario

    shifting_results = scenarios["shifting"]

    results_summary = {
        "static_seeks": shifting_results["static_seeks"],
        "learned_seek": shifting_results["learned_seek"],
        "history": shifting_results["history"],
        "scenarios": scenarios,
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    scenario_names = list(scenarios)
    algorithms = ["FCFS", "SSTF", "SCAN", "CSCAN", "Learned"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    positions = np.arange(len(scenario_names))
    bar_width = 0.15

    plt.figure(figsize=(10, 5))
    for index, (algorithm, color) in enumerate(zip(algorithms, colors)):
        costs = [
            scenario["learned_seek"]
            if algorithm == "Learned"
            else scenario["static_seeks"][algorithm]
            for scenario in scenarios.values()
        ]
        offset = (index - (len(algorithms) - 1) / 2) * bar_width
        plt.bar(positions + offset, costs, bar_width, label=algorithm, color=color)
    plt.xticks(positions, scenario_names)
    plt.ylabel("Total Seek Distance")
    plt.title("Disk Scheduling Seek Distance by Workload")
    plt.legend()
    plt.savefig("results/fig1_cumulative_excess.png")
    plt.close()

    print("Experiment execution complete. Metrics and visual figures exported to results/")


if __name__ == "__main__":
    main()