import json

import matplotlib.pyplot as plt

from src.workloads import generate_shifting_trace
from src.simulate import run_shifting_experiment


def main():
    trace = generate_shifting_trace(total_length=200, seed=42)
    static_results, learned_result, history = run_shifting_experiment(
        trace, window_size=10
    )

    results_summary = {
        "static_seeks": static_results,
        "learned_seek": learned_result,
        "history": history,
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    algs = list(static_results.keys()) + ["Learned"]
    costs = list(static_results.values()) + [learned_result]

    plt.figure(figsize=(8, 5))
    plt.bar(
        algs,
        costs,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    )
    plt.ylabel("Total Seek Distance")
    plt.title("Disk Scheduling Seek Time Under Workload Shift")
    plt.savefig("results/fig1_cumulative_excess.png")
    plt.close()

    print("Experiment execution complete. Metrics and visual figures exported to results/")


if __name__ == "__main__":
    main()