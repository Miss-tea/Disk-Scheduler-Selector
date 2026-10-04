import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.schedulers import fcfs, sstf, scan, cscan
from src.features import extract_window_features
from src.workloads import (
    generate_sequential_workload,
    generate_random_workload,
    generate_bursty_workload,
)


class SchedulerSelector:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.classes = ["FCFS", "SSTF", "SCAN", "CSCAN"]

    def generate_training_data(self, num_samples=300, window_size=15, total_tracks=200):
        X, y = [], []
        for i in range(num_samples):
            head = np.random.randint(0, total_tracks)
            w_type = i % 3
            if w_type == 0:
                reqs = generate_sequential_workload(
                    window_size,
                    total_tracks,
                    start=np.random.randint(0, 100),
                )
            elif w_type == 1:
                reqs = generate_random_workload(window_size, total_tracks)
            else:
                reqs = generate_bursty_workload(window_size, total_tracks)

            feats = extract_window_features(reqs, head, total_tracks)
            costs = {
                "FCFS": fcfs(head, reqs, total_tracks)[0],
                "SSTF": sstf(head, reqs, total_tracks)[0],
                "SCAN": scan(head, reqs, total_tracks)[0],
                "CSCAN": cscan(head, reqs, total_tracks)[0],
            }
            best_alg = min(costs, key=costs.get)

            X.append(feats)
            y.append(best_alg)

        return np.array(X), np.array(y)

    def train(self):
        X, y = self.generate_training_data()
        self.model.fit(X, y)

    def predict(self, window, head, total_tracks=200):
        feats = extract_window_features(window, head, total_tracks).reshape(1, -1)
        pred = self.model.predict(feats)[0]
        probs = self.model.predict_proba(feats)[0]
        confidence = float(np.max(probs))
        return pred, confidence