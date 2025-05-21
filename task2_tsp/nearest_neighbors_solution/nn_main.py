from gps.gps_objects import Point
from typing import List
from task2_tsp.tsp_utils import distance
import numpy as np


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


def randomized_nn(dist_matrix, start, end=None, k=5):
    N = len(dist_matrix)
    unvisited = set(range(N)) - {start}
    if end is not None:
        unvisited -= {end}

    route = [start]
    current = start

    while unvisited:
        # sort unvisited by distance from current
        unvisited_list = list(unvisited)
        dists = dist_matrix[current, unvisited_list]
        idx_sorted = np.argsort(dists)

        # take up to k nearest
        topk = idx_sorted[:min(k, len(idx_sorted))]
        # pick one at random
        choice = np.random.choice(topk)
        next_node = unvisited_list[choice]

        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    if end is not None:
        route.append(end)
    return np.array(route, dtype=int)


def two_opt_candidate_list(route: np.ndarray,
                           dist_matrix: np.ndarray,
                           K: int = 20,
                           max_passes: int = 5) -> np.ndarray:
    """
    A much faster 2-opt that only considers swapping with each node's
    K nearest neighbors.  Stops after `max_passes` over the tour
    without any improvement.

    route:     1D array of node indices, with route[0] and route[-1] fixed.
    dist_matrix: 2D numpy array of pairwise distances.
    K:         how many nearest neighbors to keep per node.
    max_passes: stop after this many consecutive passes with no improvements.
    """
    n = len(route)
    best = route.copy()
    D = dist_matrix
    # build a position lookup so we can turn city→its index in 'best'
    pos = np.empty(n, dtype=int)
    pos[best] = np.arange(n)

    # Precompute for each city the list of its K nearest neighbors (excluding itself)
    # shape (n, K)
    neighbors = np.argsort(D, axis=1)[:, 1: K + 1]

    no_improve = 0
    while no_improve < max_passes:
        improved = False
        # for each “breakpoint” i, keep 0 and n-1 fixed
        for i in range(1, n - 2):
            a = best[i - 1]
            b = best[i]
            # only try to connect a → c where c is among a's K nearest
            for c in neighbors[a]:
                j = pos[c]
                # j must be at least i+1, and j+1 < n
                if j <= i or j >= n - 1:
                    continue
                d = best[j + 1]
                # compute the delta of swapping edges (a–b, c–d) → (a–c, b–d)
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < 0:
                    # perform the 2-opt reversal
                    best[i: j + 1] = best[i: j + 1][::-1]
                    # update positions for the reversed block
                    pos[best[i: j + 1]] = np.arange(i, j + 1)
                    improved = True
                    break  # break out of neighbor loop
            if improved:
                break  # restart from i=1 on the improved tour

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return best
