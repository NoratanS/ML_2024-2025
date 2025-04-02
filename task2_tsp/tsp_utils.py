import random
from math import sqrt
from itertools import permutations

from gps.gps_objects import Point
from typing import List


def generate_random_points(n: int, x_bound: int, y_bound: int) -> List[Point]:
    """Returns n number of random points starting from (0, 0) to (x_bound, y_bound)."""
    points = set()
    while len(points) < n:
        points.add(Point(random.randint(0, x_bound + 1), random.randint(0, y_bound + 1)))

    return list(points)


def choose_random_start_end(points: List[Point]) -> (Point, Point):
    """Returns tuple of random start and end points."""
    start = random.choice(points)
    end = random.choice([p for p in points if p != start])
    return start, end


def generate_random_problem(points: List[Point], k: int) -> (Point, List[Point], Point):
    if k > len(points) - 2:
        raise ValueError('k must be less than the number of points')

    start = random.choice(points)
    points.remove(start)

    end = random.choice(points)
    points.remove(end)

    waypoints = []
    for _ in range(k):
        point = random.choice(points)
        waypoints.append(point)
        points.remove(point)

    return start, waypoints, end


def distance(point1: Point, point2: Point) -> float:
    """Returns distance between two points."""
    return round(sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2), 2)


def total_route_length(route: List[Point]) -> float:
    """Returns total route length."""
    return round(sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)), 2)


def brute_force(start: Point, waypoints: List[Point], end: Point) -> List[Point]:
    min_path_len = float('inf')
    best_path = []

    for perm in permutations(waypoints):
        current_path = [start] + list(perm) + [end]
        current_path_len = total_route_length(current_path)
        if current_path_len < min_path_len:
            min_path_len = current_path_len
            best_path = current_path

    return best_path
