from src.schedulers import fcfs, sstf, scan, cscan


def test_fcfs():
    seek, path = fcfs(50, [100, 20, 30])
    assert seek == (50 + 80 + 10)
    assert path == [50, 100, 20, 30]


def test_sstf():
    seek, path = sstf(50, [180, 52, 10])
    assert path == [50, 52, 10, 180]


def test_scan():
    seek, path = scan(50, [10, 80, 120], total_tracks=200, direction="up")
    assert path == [50, 80, 120, 199, 10]


def test_cscan():
    seek, path = cscan(50, [10, 80, 120], total_tracks=200)
    assert path == [50, 80, 120, 199, 0, 10]