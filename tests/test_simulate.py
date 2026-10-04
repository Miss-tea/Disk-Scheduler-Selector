import src.simulate as simulation


def test_simulation_processes_final_window_and_carries_head(monkeypatch):
    selectors = []

    class FakeSelector:
        def __init__(self):
            self.prediction_heads = []
            selectors.append(self)

        def train(self):
            pass

        def predict(self, window, head, total_tracks):
            self.prediction_heads.append(head)
            return "SSTF", 0.75

    monkeypatch.setattr(simulation, "SchedulerSelector", FakeSelector)

    static, learned, history = simulation.run_shifting_experiment(
        [10, 80, 120, 20, 30], window_size=3
    )

    assert len(history) == 2
    assert [entry["step"] for entry in history] == [0, 3]
    assert selectors[0].prediction_heads == [50, 10]
    assert all("oracle_best_alg" in entry and "correct" in entry for entry in history)
    assert set(static) == {"FCFS", "SSTF", "SCAN", "CSCAN"}
    assert learned >= 0