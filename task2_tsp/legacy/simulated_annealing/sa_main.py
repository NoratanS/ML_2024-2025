import numpy as np

from task2_tsp.legacy.simulated_annealing.sa_utils import generate_neighbor, get_temp_scheduler \
    , randomized_nn, two_opt_candidate_list, \
    generate_neighbor_mix_numpy_experimental
from task2_tsp.tsp_utils import total_route_len_dist_mat
import random
import math
from tqdm import tqdm


def simulated_annealing_fixed_ends(dist_matrix, start=0, end=None,
                                   initial_temp=1000.0, cooling_rate=0.999,
                                   stop_temp=1e-3, max_iter=500000):
    N = len(dist_matrix)
    if end is None:
        end = N - 1

    middle = list(range(N))
    middle.remove(start)
    middle.remove(end)
    random.shuffle(middle)

    current_route = [start] + middle + [end]
    current_cost = total_route_len_dist_mat(current_route, dist_matrix)
    best_route = current_route[:]
    best_cost = current_cost

    temp = initial_temp
    iteration = 0

    while iteration < max_iter:
        new_route = generate_neighbor(current_route)
        new_cost = total_route_len_dist_mat(new_route, dist_matrix)

        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_route = new_route
            current_cost = new_cost

            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost

        if temp > stop_temp:
            temp *= cooling_rate
        else:
            temp = stop_temp
        iteration += 1

    return best_route, best_cost


def simulated_annealing_experimental(dist_matrix, start=0, end=None,
                                     initial_temp=1000.0, cooling_rate=0.995,
                                     stop_temp=1e-3, max_iter=100000,
                                     cooling_scheme="exponential",
                                     verbose=False, show_progress=True):
    N = len(dist_matrix)
    if end is None:
        end = N - 1

    # seed
    current_route = randomized_nn(dist_matrix, start=0, end=N - 1, k=5)
    # fast 2-opt
    current_route = two_opt_candidate_list(current_route, dist_matrix, K=20, max_passes=3)
    current_cost = total_route_len_dist_mat(current_route, dist_matrix)
    best_route = current_route[:]
    best_cost = current_cost

    # Select temperature scheduler
    scheduler = get_temp_scheduler(cooling_scheme, initial_temp, cooling_rate)

    K = 20
    neighbor_list = np.argsort(dist_matrix, axis=1)[:, 1:K + 1]

    progress_iter = tqdm(range(max_iter), disable=not show_progress, desc="Simulated Annealing")

    for iteration in progress_iter:
        temp = scheduler(iteration)
        if temp < stop_temp:
            break

        new_route = generate_neighbor_mix_numpy_experimental(
            current_route,
            dist_matrix,
            temp,
            initial_temp,
            neighbor_list,
            p_2opt=0.5,
            p_oropt=0.2,
            p_swap=0.2,
            p_reloc=0.1
        )

        new_cost = total_route_len_dist_mat(new_route, dist_matrix)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_route = new_route
            current_cost = new_cost
            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost

        if verbose and iteration % 1000 == 0:
            progress_iter.set_postfix(cost=best_cost, temp=temp)

    return best_route, best_cost
