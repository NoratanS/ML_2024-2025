from task2_tsp.nearest_neighbors_solution.nn_main import nearest_neighbors
from task2_tsp.legacy.simulated_annealing.sa_utils import iterated_local_search
from task2_tsp.tsp_utils import generate_random_points, generate_random_problem, total_route_length, euclidean_distance_matrix

n = 10000
k = 971
x_bound = 5000
y_bound = 5000
test_arr = generate_random_points(n, x_bound, y_bound)
start, waypoints, end = generate_random_problem(test_arr, k)
print(start, waypoints, end)
print("\n")
points = [start] + waypoints + [end]
distances = euclidean_distance_matrix(points)


"""best_route, best_cost = simulated_annealing_fixed_ends(distances, start=0, end=len(points)-1)
print("Best route:", [points[i] for i in best_route])
print("Total cost:", best_cost)"""

"""best_route, best_cost = simulated_annealing_experimental(
    distances,
    start=0,
    end=len(points) - 1,
    initial_temp=60000,
    stop_temp=10,
    cooling_rate=0.9,
    max_iter=200000,
    cooling_scheme="logarithmic",
    verbose=True,
    show_progress=True
)

print("Best route:", [points[i] for i in best_route])
print("Total cost:", best_cost)"""


"""r = randomized_nn(distances, start=0, end=k-1, k=5)
r = two_opt_candidate_list(r, distances, K=20, max_passes=3)  # your fast 2-opt
r_cost = total_route_len_dist_mat(r, distances)
print(r_cost)


best_path_nn = nearest_neighbors(start, waypoints, end)
print(best_path_nn, '\n', total_route_length(best_path_nn))

killer_route = killer_initial_route(distances)
print('\n', total_route_len_dist_mat(route=killer_route, dist_mat=distances))"""

best_path_nn = nearest_neighbors(start, waypoints, end)
print(best_path_nn, '\n', total_route_length(best_path_nn))

r_b, r_b_c = iterated_local_search(distances)
print(r_b, r_b_c)




"""best_path_ga = best_route = genetic_algorithm(
    start=start,
    end=end,
    waypoints=waypoints,
    generations=300,
    pop_size=100,
    initial_mutation_rate=0.1,
    elite_size=10,
    stagnation_limit=10,
    mutation_rate_bounds=(0.1, 0.1),
    mutation_step=0.03
)
print('\n')
print(best_path_ga, '\n', total_route_length(best_path_ga))"""

