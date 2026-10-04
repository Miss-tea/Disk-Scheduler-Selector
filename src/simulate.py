from src.schedulers import fcfs, sstf, scan, cscan
from src.selector import SchedulerSelector


def run_shifting_experiment(trace, window_size=10, total_tracks=200):
    selector = SchedulerSelector()
    selector.train()

    head_learned = 50
    heads_static = {"FCFS": 50, "SSTF": 50, "SCAN": 50, "CSCAN": 50}

    total_seeks_static = {"FCFS": 0, "SSTF": 0, "SCAN": 0, "CSCAN": 0}
    total_seek_learned = 0

    history = []

    for i in range(0, len(trace), window_size):
        window = trace[i : i + window_size]
        alg_map = {"FCFS": fcfs, "SSTF": sstf, "SCAN": scan, "CSCAN": cscan}
        static_window_costs = {}

        for alg, fn in alg_map.items():
            seek, path = fn(heads_static[alg], window, total_tracks)
            total_seeks_static[alg] += seek
            heads_static[alg] = path[-1]
            static_window_costs[alg] = seek

        chosen_alg, confidence = selector.predict(window, head_learned, total_tracks)
        window_costs = {
            alg: fn(head_learned, window, total_tracks)[0]
            for alg, fn in alg_map.items()
        }
        oracle_best_alg = min(window_costs, key=window_costs.get)

        seek, path = alg_map[chosen_alg](head_learned, window, total_tracks)
        total_seek_learned += seek
        head_learned = path[-1]

        history.append(
            {
                "step": i,
                "chosen_alg": chosen_alg,
                "confidence": confidence,
                "seek_cost": seek,
                "static_costs": static_window_costs,
                "window_costs": window_costs,
                "oracle_best_alg": oracle_best_alg,
                "correct": chosen_alg == oracle_best_alg,
            }
        )

    return total_seeks_static, total_seek_learned, history