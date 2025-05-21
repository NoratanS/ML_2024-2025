import random
import numpy as np
from math import sqrt
from itertools import permutations

from gps.gps_objects import Point
from typing import List


def points_to_numpy(points: List[Point]):
    """Convert list of points to Nx2 numpy array"""
    return np.array([[p.x, p.y] for p in points])


def euclidean_distance_matrix(points: List[Point]) -> np.ndarray:
    """generate euclidean distance matrix
        use the points list to get indices, for example:
        if you have points: [(0, 0), (3, 4), (6,8)] to calculate route from start to end your route would be [0, 1, 2]
        if you want to go from end to start your route would be [2, 1, 0]
        the distance from 0 to 1 is dist[0, 1]"""
    coords = points_to_numpy(points)  # shape: (N, 2)
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))  # (N, N)
    return dists


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


def total_route_len_dist_mat(route: List[int], dist_mat: np.ndarray) -> float:
    """Returns total route length using distance matrix."""
    return sum(dist_mat[route[i], route[i + 1]] for i in range(len(route) - 1))


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


def load_points(file_path: str) -> List[Point]:
    """
    Reads a file where each non-empty line contains 'x%y',
    parses the coordinates, and returns a list of Point objects.
    """
    points: List[Point] = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split('%')
            x = int(float(x_str.strip()) * 100)
            y = int(float(y_str.strip()) * 100)
            points.append(Point(x, y))
    return points