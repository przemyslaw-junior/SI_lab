import random
import copy
import matplotlib.pyplot as plt
import numpy as np

# Liczba hetmanów / rozmiar planszy
# Generowanie pojedynczego osobnika
def generate_individual(board_size):
    individual = []
    for i in range(board_size):
# Generowanie losowej pozycji hetmana
        pos_x = random.randint(1, board_size)
        pos_y = random.randint(1, board_size)
        individual.append((pos_x, pos_y))
    return individual

# Generowanie początkowej populacji osobników
def generate_population(population_size, board_size):
    population = []
    for i in range(population_size):
        individual = generate_individual(board_size)
        population.append(individual)
    return population

# Funkcja przystosowania -> oblicza liczbę ataków między hetmanami w danym osobniku
def fitness(individual):
    attacks = 0
    num_queens = len(individual)
# Porównanie hetmanów
    for i in range(num_queens):
        for j in range(i + 1, num_queens):
            x1, y1 = individual[i]
            x2, y2 = individual[j]
# Ataki Hetmanów
            if x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2):
                attacks += 1
    return attacks

# Ocena populacji -> oblicza funkcję przystosowania dla każdego osobnika
def evaluate_population(population):
    fitness_list = []
    for individual in population:
        fitness_value = fitness(individual)
        fitness_list.append(fitness_value)
    return fitness_list

# Selekcja turniejowa -> z losowej próbki osobników wybiera tego z najlepszym (najniższym) wynikiem przystosowania
def tournament_selection(population, fitness_list, tournament_size):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    best_index = tournament_indices[0]
    for index in tournament_indices:
        if fitness_list[index] < fitness_list[best_index]:
            best_index = index
# Zwracamy kopię wybranego osobnika, aby nie modyfikować oryginału
    return copy.deepcopy(population[best_index])

# Tworzenie nowej populacji przy użyciu selekcji turniejowej
def selection(population, tournament_size):
    new_population = []
    fitness_list = evaluate_population(population)
    for i in range(len(population)):
        selected_individual = tournament_selection(population, fitness_list, tournament_size)
        new_population.append(selected_individual)
    return new_population

# Krzyżowanie jednopunktowe -> wymiana fragmentów między dwoma rodzicami w celu stworzenia potomstwa
def crossover(population, crossover_probability, board_size):
    population_size = len(population)
    i = 0
# Przetwarzanie osobników w parach
    while i < population_size - 1:
        if random.random() < crossover_probability:
            parent1 = population[i]
            parent2 = population[i + 1]
            if board_size > 1:
# Wybór losowego punkt krzyżowania między 1 a board_size-1
                crossover_point = random.randint(1, board_size - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                population[i] = child1
                population[i + 1] = child2
        i += 2
    return population

# Mutacja -> losowa zmiana jednej współrzędnej w losowo wybranym genie osobnika
def mutation(population, mutation_probability, board_size):
    for individual in population:
        if random.random() < mutation_probability:
            gene_index = random.randint(0, board_size - 1)
            x, y = individual[gene_index]
# Losowo decydujemy, czy mutować współrzędną x czy y
            if random.random() < 0.5:
                new_x = random.randint(1, board_size)
                individual[gene_index] = (new_x, y)
            else:
                new_y = random.randint(1, board_size)
                individual[gene_index] = (x, new_y)
    return population

# Główna funkcja algorytmu ewolucyjnego
def evolutionary_algorithm(board_size, population_size, tournament_size, max_generations, crossover_probability, mutation_probability):
    population = generate_population(population_size, board_size)
    fitness_list = evaluate_population(population)
    best_fitness = min(fitness_list)
    best_individual = population[fitness_list.index(best_fitness)]
    
    best_history = []
    avg_history = []
    generation = 0
    
    # Aalgorytm działa aż znajdzie rozwiązanie (0 ataków) lub osiągnięto maksymalną liczbę generacji
    while generation < max_generations and best_fitness > 0:
        best_history.append(best_fitness)
        avg_history.append(sum(fitness_list) / len(fitness_list))
        
        population = selection(population, tournament_size)
        population = crossover(population, crossover_probability, board_size)
        population = mutation(population, mutation_probability, board_size)
        
        fitness_list = evaluate_population(population)
        current_best = min(fitness_list)
        if current_best < best_fitness:
            best_fitness = current_best
            best_individual = population[fitness_list.index(best_fitness)]
            
        generation += 1
    
    best_history.append(best_fitness)
    avg_history.append(sum(fitness_list) / len(fitness_list))
    
    return best_individual, best_fitness, best_history, avg_history, generation

# Wyświetlanie planszy
def print_board(individual, board_size):
    board = [['.' for _ in range(board_size)] for _ in range(board_size)]
    for (x, y) in individual:
        board[x - 1][y - 1] = 'Q'
    for row in board:
        print(" ".join(row))

# Rysowanie wykresów
def plot_history(best_history, avg_history):
    generations = range(len(best_history))
    plt.plot(generations, best_history, label="Najlepsze przystosowanie")
    plt.plot(generations, avg_history, label="Średnie przystosowanie")
    plt.xlabel("Generacja")
    plt.ylabel("Liczba ataków")
    plt.title("Ewolucja przystosowania")
    plt.legend()
    plt.grid(True)
    plt.show()

# Główna część programu
if __name__ == "__main__":
    board_size = 8              
    population_size = 10      
    tournament_size = 5         
    max_generations = 1000       
    crossover_probability = 0.7
    mutation_probability = 0.8
    
    best_individual, best_fitness, best_history, avg_history, generations = evolutionary_algorithm(
        board_size, population_size, tournament_size, max_generations, crossover_probability, mutation_probability)
    
    print("Najlepsze rozwiązanie:")
    print_board(best_individual, board_size)
    print("Liczba ataków:", best_fitness)
    print("Generacje:", generations)
    
    plot_history(best_history, avg_history)