import random


def create_individual(waypoints):
    """Losowa permutacja punktów pośrednich."""
    individual = waypoints[:]
    random.shuffle(individual)
    return individual


def crossover(parent1, parent2):
    """Crossover z pełnym sprawdzeniem — podobny do PMX."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    # Kopiujemy fragment z parent1
    child[start:end] = parent1[start:end]

    # Uzupełniamy resztę genami z parent2 w kolejności, jeśli ich nie ma już w child
    current_index = 0
    for gene in parent2:
        if gene not in child:
            while child[current_index] is not None:
                current_index += 1
            if current_index < size:
                child[current_index] = gene

    # Na koniec: kontrola bezpieczeństwa
    if None in child:
        raise ValueError("Crossover failed: Child contains None.")

    return child


def mutate(individual, mutation_rate=0.1):
    """Zamiana dwóch punktów miejscami z pewnym prawdopodobieństwem."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]


def select(population, fitnesses, num):
    """Selekcja najlepszych osobników."""
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])]
    return sorted_population[:num]