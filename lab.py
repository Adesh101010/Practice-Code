import random

# Define the parameters
target_sum = 30
population_size = 57
mutation_rate = 0.2
num_generations = 10

# Define the function to maximize
def fitness_function(individual):
    a, b, c, d = individual
    return a + 2*b + 3*c + 4*d

# Generate initial population
def generate_individual():
    return [random.randint(0, target_sum) for _ in range(4)]

def generate_population():
    return [generate_individual() for _ in range(population_size)]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, target_sum)
    return individual

# Evolution
def evolve(population):
    graded = [(fitness_function(individual), individual) for individual in population]
    graded = sorted(graded, key=lambda x: x[0], reverse=True)
    parents = [individual for _, individual in graded[:int(0.2 * len(graded))]]

    # Keep top individuals, perform crossover and mutation
    children = []
    while len(children) < len(population) - len(parents):
        parent1, parent2 = random.sample(parents, 2)
        child1, child2 = crossover(parent1, parent2)
        children.append(mutate(child1))
        children.append(mutate(child2))
    parents.extend(children)
    return parents

# Main
population = generate_population()

for generation in range(num_generations):
    population = evolve(population)
    best_individual = max(population, key=fitness_function)
    print(f"Generation {generation+1}: Best individual = {best_individual}, Fitness = {fitness_function(best_individual)}")
