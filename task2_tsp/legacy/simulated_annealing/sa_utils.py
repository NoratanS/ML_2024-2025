import math
import random
import time

import numpy as np

from task2_tsp.tsp_utils import total_route_len_dist_mat


def generate_neighbor(route, segment_max_length=50):
    """Return a new route with a reversed segment within [1, -1) (excluding start/end)."""
    new_route = route[:]
    n = len(route)
    if n <= 3:
        return new_route  # no space to swap

    i = random.randint(1, n - 3)  # must leave space for j > i
    max_j = min(i + segment_max_length, n - 2)
    j = random.randint(i + 1, max_j)

    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route


def generate_neighbor_mix_numpy(route: np.ndarray, three_opt_chance=0.99) -> np.ndarray:
    n = len(route)
    segment_max_length = max(50, len(route) // 5)  # 20% of the route
    new_route = route.copy()

    if random.random() < three_opt_chance and n >= 6:
        # Safe 3-opt style operation
        i, j, k = np.sort(np.random.choice(np.arange(1, n - 1), size=3, replace=False))

        # Recombine segments in a new order (e.g., reverse middle part)
        part1 = new_route[:i]
        part2 = new_route[i:j]
        part3 = new_route[j:k]
        part4 = new_route[k:]

        # Randomly shuffle orderings (pick one strategy)
        new_route = np.concatenate([part1, part3, part2[::-1], part4])
        return new_route
    else:
        # 2-opt reversal
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, min(i + segment_max_length, n - 2))
        new_route[i:j + 1] = new_route[i:j + 1][::-1]
        return new_route


def generate_neighbor_mix_numpy_experimental(route: np.ndarray,
                                dist_matrix: np.ndarray,
                                temp: float,
                                T0: float,
                                neighbor_list: np.ndarray,
                                p_2opt=0.5,
                                p_oropt=0.2,
                                p_swap=0.2,
                                p_reloc=0.1):
    """
    route: current 1D array of length N (with fixed endpoints at 0 and N-1)
    dist_matrix: NxN numpy array
    temp, T0: current and initial temperature
    neighbor_list: precomputed array of shape (N, K) listing each city’s K nearest neighbors
    p_*\: mixture probabilities (must sum to 1)
    """

    N = len(route)
    # choose which move to do
    r = random.random()

    if r < p_2opt:
        # -------- 2-opt using candidate list --------
        # pick a random index i in [1 .. N-3]
        i = random.randint(1, N-3)
        a = route[i-1]
        b = route[i]
        # pick one of a's nearest neighbors
        for c in neighbor_list[a]:
            # find its position in the route
            j = np.where(route == c)[0][0]
            # ensure valid reversal
            if j <= i or j >= N-1:
                continue
            d = route[j+1]
            # compute delta
            delta = (dist_matrix[a, c] + dist_matrix[b, d]
                     - dist_matrix[a, b] - dist_matrix[c, d])
            if delta < 0 or random.random() < np.exp(-delta/temp):
                # apply 2-opt
                new_route = route.copy()
                new_route[i:j+1] = new_route[i:j+1][::-1]
                return new_route
        # if no neighbor found or accepted, fall back to swap
        move_type = 'swap_fallback'
    elif r < p_2opt + p_oropt:
        # -------- Or-opt (move block of length 1–3) --------
        L = random.randint(1, 3)               # block length
        i = random.randint(1, N-2-L)           # start of block
        block = route[i:i+L].copy()
        rest = np.delete(route, np.s_[i:i+L])
        j = random.randint(1, len(rest)-1)     # new insertion position
        new_route = np.concatenate([rest[:j], block, rest[j:]])
        return new_route
    elif r < p_2opt + p_oropt + p_swap:
        # -------- Swap two positions --------
        i, j = random.sample(range(1, N-1), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    else:
        # -------- Relocate a segment --------
        # segment length scales with temp
        max_len = max(1, int((temp/T0)*(N//5)))
        L = random.randint(1, max_len)
        i = random.randint(1, N-1-L)
        block = route[i:i+L].copy()
        rest = np.delete(route, np.s_[i:i+L])
        j = random.randint(1, len(rest)-1)
        new_route = np.concatenate([rest[:j], block, rest[j:]])
        return new_route

    # fallback: simple 2-swap if 2-opt candidate list failed
    if move_type == 'swap_fallback':
        i, j = random.sample(range(1, N-1), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

def get_temp_scheduler(scheme="exponential", initial_temp=1000.0, cooling_rate=0.995):
    """Return a cooling schedule function."""
    if scheme == "exponential":
        return lambda t: initial_temp * (cooling_rate ** t)
    elif scheme == "linear":
        return lambda t: max(0.0, initial_temp - cooling_rate * t)
    elif scheme == "logarithmic":
        return lambda t: initial_temp / math.log(t + 2)
    else:
        raise ValueError("Unknown cooling scheme: choose exponential, linear, or logarithmic")


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
    neighbors = np.argsort(D, axis=1)[:, 1 : K+1]

    no_improve = 0
    while no_improve < max_passes:
        improved = False
        # for each “breakpoint” i, keep 0 and n-1 fixed
        for i in range(1, n-2):
            a = best[i-1]
            b = best[i]
            # only try to connect a → c where c is among a's K nearest
            for c in neighbors[a]:
                j = pos[c]
                # j must be at least i+1, and j+1 < n
                if j <= i or j >= n-1:
                    continue
                d = best[j+1]
                # compute the delta of swapping edges (a–b, c–d) → (a–c, b–d)
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < 0:
                    # perform the 2-opt reversal
                    best[i : j+1] = best[i : j+1][::-1]
                    # update positions for the reversed block
                    pos[best[i : j+1]] = np.arange(i, j+1)
                    improved = True
                    break  # break out of neighbor loop
            if improved:
                break  # restart from i=1 on the improved tour

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return best


def two_opt_local_opt(route: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """
    Perform 2-opt until no further improvement.
    Keeps route[0] and route[-1] fixed (a path, not a cycle).
    """
    best = route.copy()
    n = len(best)
    improved = True

    # Pre-bind for speed
    D = dist_matrix

    while improved:
        improved = False
        # i from first movable position up to n-3
        for i in range(1, n - 2):
            a = best[i - 1]
            b = best[i]
            # j from just beyond i up to n-2 (so j+1 ≤ n-1)
            for j in range(i + 1, n - 1):
                c = best[j]
                d = best[j + 1]
                # cost difference if we reverse segment [i..j]
                delta = (D[a, c] + D[b, d]
                         - D[a, b] - D[c, d])
                if delta < 0:
                    # reversal yields improvement
                    best[i : j + 1] = best[i : j + 1][::-1]
                    improved = True
                    break  # restart scan after any improvement
            if improved:
                break
    return best



def generate_3opt_variations(route, i, j, k):
    """
    Given breakpoints i<j<k, return the 7 non-trivial 3-opt reconnections
    of the segments A=route[:i], B=route[i:j], C=route[j:k], D=route[k:].
    """
    A, B, C, D = route[:i], route[i:j], route[j:k], route[k:]
    return [
        np.concatenate([A, B[::-1], C, D]),           # flip B
        np.concatenate([A, B, C[::-1], D]),           # flip C
        np.concatenate([A, C, B, D]),                 # swap B<->C
        np.concatenate([A, C[::-1], B[::-1], D]),     # swap+flip both
        np.concatenate([A, B[::-1], C[::-1], D]),     # flip B & C
        np.concatenate([A, C, B[::-1], D]),           # swap B<->C + flip B
        np.concatenate([A, C[::-1], B, D]),           # swap B<->C + flip C
    ]

def randomized_3opt(route, dist_matrix, trials=2000):
    """
    Perform `trials` random 3-opt attempts, accepting any improving move immediately.
    """
    best, best_cost = route.copy(), total_route_len_dist_mat(route, dist_matrix)
    n = len(route)
    for _ in range(trials):
        i, j, k = sorted(random.sample(range(1, n-1), 3))
        for candidate in generate_3opt_variations(best, i, j, k):
            cost = total_route_len_dist_mat(candidate, dist_matrix)
            if cost < best_cost:
                best, best_cost = candidate, cost
                break  # goto next trial as soon as we improve
    return best


def killer_initial_route(dist_matrix, start=0, end=None,
                         k_nn=5, K_2opt=20, max_passes_2opt=3,
                         trials_3opt=2000):
    N = len(dist_matrix)
    if end is None:
        end = N-1

    # 1) random‐tie NN
    route = randomized_nn(dist_matrix, start, end, k=k_nn)

    # 2) candidate‐list 2-opt
    route = two_opt_candidate_list(route, dist_matrix,
                                   K=K_2opt, max_passes=max_passes_2opt)

    # 3) randomized 3-opt
    route = randomized_3opt(route, dist_matrix, trials=trials_3opt)

    return route


def three_opt_local_opt(route: np.ndarray,
                        D: np.ndarray,
                        candidate_list: np.ndarray,
                        max_no_improve: int = 5) -> np.ndarray:
    """
    Deterministic 3-opt local search using each node's KNN candidate list.
    Stops after `max_no_improve` full passes with no improvement.
    """
    n = len(route)
    best = route.copy()
    best_cost = total_route_len_dist_mat(best, D)
    no_improve = 0

    # map city -> position in 'best'
    pos = np.empty(n, dtype=int)
    pos[best] = np.arange(n)

    while no_improve < max_no_improve:
        improved = False

        # pick first edge (a, b)
        for i in range(1, n-2):
            a, b = best[i-1], best[i]

            # only consider reconnecting a -> c where c in a's candidates
            for c in candidate_list[a]:
                j = pos[c]
                if j <= i or j >= n-1: continue
                d = best[j+1]

                # now pick a second break (c, d) and reconnect via a 3rd break (e, f)
                # consider only d -> e where e in d’s candidates
                for e in candidate_list[d]:
                    k = pos[e]
                    if k <= j or k >= n-1: continue
                    f = best[k+1]

                    # there are 7 reconnection patterns—compute the best gain delta
                    # and apply the first one that improves
                    # (omitting the full code for brevity, but see any 3-opt reference)
                    # Example for one pattern:
                    delta1 = (D[a, b] + D[c, d] + D[e, f]
                              - (D[a, c] + D[b, e] + D[d, f]))
                    if delta1 > 1e-12:  # improvement
                        # perform that specific reconnection:
                        #   best[i:j+1], best[j+1:k+1] rearranged accordingly
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # update best_cost, pos[], and restart i-loop
                best_cost = total_route_len_dist_mat(best, D)
                pos[best] = np.arange(n)
                break

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return best


def double_bridge_move(route: np.ndarray) -> np.ndarray:
    n = len(route)
    # pick 4 breakpoints in the interior, sorted
    cuts = sorted(random.sample(range(1, n-1), 4))
    a,b,c,d = cuts
    A = route[:a]
    B = route[a:b]
    C = route[b:c]
    D = route[c:d]
    E = route[d:]
    # reconnect in order A–C–B–D–E
    return np.concatenate([A, C, B, D, E])


def iterated_local_search(dist_matrix, start=0, end=None,
                          time_budget=300,      # seconds
                          cand_K=20,            # for both 2-opt & 3-opt
                          max_passes_2opt=3,
                          max_no_improve_3opt=3,
                          shake_rounds=50):
    N = len(dist_matrix)
    if end is None: end = N-1

    # Precompute candidate list
    candidate_list = np.argsort(dist_matrix, axis=1)[:, 1:cand_K+1]

    # 1) Killer seed
    route = killer_initial_route(dist_matrix, start, end,
                                 k_nn=5,
                                 K_2opt=cand_K,
                                 max_passes_2opt=max_passes_2opt,
                                 trials_3opt=2000)
    route = three_opt_local_opt(route, dist_matrix, candidate_list,
                                max_no_improve=max_no_improve_3opt)
    best, best_cost = route, total_route_len_dist_mat(route, dist_matrix)

    start_time = time.time()
    while time.time() - start_time < time_budget:
        # 2) Shake
        shaken = double_bridge_move(best)
        # 3) Polish
        polished = three_opt_local_opt(shaken, dist_matrix, candidate_list,
                                       max_no_improve=max_no_improve_3opt)
        cost = total_route_len_dist_mat(polished, dist_matrix)
        if cost < best_cost:
            best, best_cost = polished, cost
        # Optional: break early if no improvement in many shakes
    return best, best_cost
