from task2_tsp.nearest_neighbors_solution.nn_main import nearest_neighbors
from task2_tsp.tsp_utils import generate_random_points, generate_random_problem, brute_force, total_route_length

n = 10000
k = 500
x_bound = 10000
y_bound = 10000
test_arr = generate_random_points(n, x_bound, y_bound)
start, waypoints, end = generate_random_problem(test_arr, k)

print(start, waypoints, end)
best_path_bf = brute_force(start, waypoints, end)
best_path_nn = nearest_neighbors(start, waypoints, end)

print(best_path_bf, '\n', total_route_length(best_path_bf))
print('\n')
print(best_path_nn, '\n', total_route_length(best_path_nn))