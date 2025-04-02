from gps.gps_objects import Point
from typing import List
from task2_tsp.tsp_utils import distance


def nearest_neighbors(start: Point, waypoints: List[Point], end: Point) -> List[Point]:
    current = start
    remaining = waypoints[:]
    ordered = [start]

    while remaining:
        next = min(remaining, key=lambda p: distance(current, p))
        ordered.append(next)
        remaining.remove(next)
        current = next

    ordered.append(end)
    return ordered
