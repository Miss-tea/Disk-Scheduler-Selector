def fcfs(start_head: int, requests: list[int], total_tracks: int = 200) -> tuple[int, list[int]]:
    """FCFS """
    current = start_head
    total_seek = 0
    path = [current]
    for req in requests:
        total_seek += abs(req - current)
        current = req
        path.append(current)
    return total_seek, path


def sstf(start_head: int, requests: list[int], total_tracks: int = 200) -> tuple[int, list[int]]:
    """SSTF"""
    current = start_head
    total_seek = 0
    pending = list(requests)
    path = [current]
    while pending:
        closest = min(pending, key=lambda x: abs(x - current))
        total_seek += abs(closest - current)
        current = closest
        path.append(current)
        pending.remove(closest)
    return total_seek, path


def scan(
    start_head: int,
    requests: list[int],
    total_tracks: int = 200,
    direction: str = "up",
) -> tuple[int, list[int]]:
    """SCAN"""
    current = start_head
    total_seek = 0
    path = [current]
    reqs = sorted(list(set(requests)))

    left = [r for r in reqs if r < start_head]
    right = [r for r in reqs if r >= start_head]

    if direction == "up":
        for r in right:
            total_seek += abs(r - current)
            current = r
            path.append(current)
        if left:
            total_seek += abs((total_tracks - 1) - current)
            current = total_tracks - 1
            path.append(current)
            for r in reversed(left):
                total_seek += abs(r - current)
                current = r
                path.append(current)
    else:
        for r in reversed(left):
            total_seek += abs(r - current)
            current = r
            path.append(current)
        if right:
            total_seek += abs(0 - current)
            current = 0
            path.append(current)
            for r in right:
                total_seek += abs(r - current)
                current = r
                path.append(current)

    return total_seek, path


def cscan(start_head: int, requests: list[int], total_tracks: int = 200) -> tuple[int, list[int]]:
    """C-SCAN"""
    current = start_head
    total_seek = 0
    path = [current]
    reqs = sorted(list(set(requests)))

    left = [r for r in reqs if r < start_head]
    right = [r for r in reqs if r >= start_head]

    for r in right:
        total_seek += abs(r - current)
        current = r
        path.append(current)

    if left:
        total_seek += abs((total_tracks - 1) - current)
        current = total_tracks - 1
        path.append(current)

        total_seek += total_tracks - 1
        current = 0
        path.append(current)

        for r in left:
            total_seek += abs(r - current)
            current = r
            path.append(current)

    return total_seek, path