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

    for i in range(0, len(trace) - window_size, window_size):
        window = trace[i : i + window_size]

        for alg, fn in [
            ("FCFS", fcfs),
            ("SSTF", sstf),
            ("SCAN", scan),
            ("CSCAN", cscan),
        ]:
            seek, _ = fn(heads_static[alg], window, total_tracks)
            total_seeks_static[alg] += seek
            heads_static[alg] = window[-1]

        chosen_alg, confidence = selector.predict(window, head_learned, total_tracks)
        alg_map = {"FCFS": fcfs, "SSTF": sstf, "SCAN": scan, "CSCAN": cscan}

        seek, _ = alg_map[chosen_alg](head_learned, window, total_tracks)
        total_seek_learned += seek
        head_learned = window[-1]

        history.append(
            {
                "step": i,
                "chosen_alg": chosen_alg,
                "confidence": confidence,
                "seek_cost": seek,
            }
        )

    return total_seeks_static, total_seek_learned, history