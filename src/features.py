import numpy as np


def extract_window_features(
    window: list[int], current_head: int, total_tracks: int = 200
) -> np.ndarray:
    """Extract variance, range, mean head distance, and direction monotonicity."""
    arr = np.array(window)
    var_tracks = float(np.var(arr))
    span = float(np.ptp(arr))
    mean_dist = float(np.mean(np.abs(arr - current_head)))

    diffs = np.diff(arr)
    if len(diffs) > 0:
        monotonicity = float(np.abs(np.sum(np.sign(diffs))) / len(diffs))
    else:
        monotonicity = 0.0

    return np.array([var_tracks, span, mean_dist, monotonicity])