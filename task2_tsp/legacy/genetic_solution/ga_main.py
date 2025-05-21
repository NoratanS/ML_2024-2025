import random
from task2_tsp.legacy.genetic_solution.ga_utils import create_individual, crossover, mutate
from task2_tsp.tsp_utils import total_route_length


def genetic_algorithm(
        start, end, waypoints,
        generations=200,
        pop_size=100,
        initial_mutation_rate=0.1,
        elite_size=5,
        stagnation_limit=15,
        mutation_rate_bounds=(0.01, 0.4),
        mutation_step=0.05
):
    # Inicjalizacja populacji
    population = [create_individual(waypoints) for _ in range(pop_size)]

    best_path = None
    best_score = float('inf')
    mutation_rate = initial_mutation_rate
    stagnation_counter = 0

    for gen in range(generations):
        # Oblicz fitness
        fitnesses = [total_route_length([start] + indiv + [end]) for indiv in population]
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])]

        current_best_score = total_route_length([start] + sorted_population[0] + [end])
        current_best_path = sorted_population[0]

        # Sprawdź poprawę
        if current_best_score < best_score:
            best_score = current_best_score
            best_path = current_best_path
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Adaptacyjna mutacja
        if stagnation_counter >= stagnation_limit:
            mutation_rate = min(mutation_rate + mutation_step, mutation_rate_bounds[1])
            stagnation_counter = 0  # resetuj licznik
        else:
            mutation_rate = max(mutation_rate - mutation_step / 2, mutation_rate_bounds[0])

        # Selekcja + elity
        elites = sorted_population[:elite_size]
        selected = sorted_population[:pop_size // 2]

        children = []
        while len(children) < (pop_size - elite_size):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            children.append(child)

        population = elites + children

        # Debug info (możesz włączyć)
        print(f"Gen {gen + 1}: Best = {best_score}, Mutation = {mutation_rate:.3f}")

    return [start] + best_path + [end]
