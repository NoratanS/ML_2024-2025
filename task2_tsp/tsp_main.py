from tsp_utils import load_points, generate_random_problem, euclidean_distance_matrix, total_route_len_dist_mat
from nearest_neighbors_solution.nn_main import randomized_nn, two_opt_candidate_list

data = load_points('data.txt')


problem_size = [158, 372, 592, 971]
for k in problem_size:
    problem_data = data.copy()
    start, waypoints, end = generate_random_problem(problem_data, k)
    del problem_data
    points = [start] + waypoints + [end]
    distances = euclidean_distance_matrix(points)

    r = randomized_nn(distances, start=0, end=k-1, k=5)
    r = two_opt_candidate_list(r, distances, K=20, max_passes=3)
    r_cost = round((total_route_len_dist_mat(r, distances) / 100), 2) # switch to meters
    print(len(data))
    print(f"k: {k}      length: {r_cost}")
