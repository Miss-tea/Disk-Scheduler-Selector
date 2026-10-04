import numpy as np


def generate_sequential_workload(length=50, total_tracks=200, start=10, stride=3):
    return [(start + i * stride) % total_tracks for i in range(length)]


def generate_random_workload(length=50, total_tracks=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [int(request) for request in np.random.randint(0, total_tracks, size=length)]


def generate_bursty_workload(length=50, total_tracks=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    clusters = [20, 100, 180]
    requests = []
    for _ in range(length):
        c = np.random.choice(clusters)
        req = int(np.clip(np.random.normal(c, 5), 0, total_tracks - 1))
        requests.append(req)
    return requests


def generate_shifting_trace(total_length=200, total_tracks=200, seed=42):
    half = total_length // 2
    w1 = generate_sequential_workload(length=half, total_tracks=total_tracks)
    w2 = generate_bursty_workload(length=half, total_tracks=total_tracks, seed=seed)
    return w1 + w2