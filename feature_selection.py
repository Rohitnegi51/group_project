import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from minisom import MiniSom
import random

def fitness_function(X, y, feature_subset):
    if sum(feature_subset) == 0:
        return 0
    selected_idx = [i for i, bit in enumerate(feature_subset) if bit == 1]
    clf = KNeighborsClassifier(n_neighbors=3)
    score = cross_val_score(clf, X[:, selected_idx], y, cv=3).mean()
    return score

# 1. Chaotic Reptile Search Algorithm (CRSA)
def reptile_search_algorithm(X, y, pop_size=10, max_iter=20):
    n_features = X.shape[1]
    def logistic_map(size, x0=0.7, r=3.9):
        seq = np.zeros(size)
        seq[0] = x0
        for i in range(1, size):
            seq[i] = r * seq[i-1] * (1 - seq[i-1])
        return seq

    chaotic_seq = logistic_map(pop_size * n_features).reshape((pop_size, n_features))
    population = (chaotic_seq > 0.5).astype(int)
    fitness = [fitness_function(X, y, ind) for ind in population]
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for iter in range(max_iter):
        for i in range(pop_size):
            new_individual = population[i].copy()
            for j in range(n_features):
                if random.random() < 0.3:
                    new_individual[j] = 1 - new_individual[j]
                elif random.random() < 0.2:
                    new_individual[j] = best_solution[j]
            new_fit = fitness_function(X, y, new_individual)
            if new_fit > fitness[i]:
                population[i] = new_individual
                fitness[i] = new_fit
                if new_fit > best_fitness:
                    best_solution = new_individual.copy()
                    best_fitness = new_fit

    selected_idx = [i for i, bit in enumerate(best_solution) if bit == 1]
    if len(selected_idx) == 0: selected_idx = [np.random.randint(n_features)]
    return selected_idx

# 2. Particle Swarm Optimization (PSO)
def pso_feature_selection(X, y, pop_size=10, max_iter=20):
    n_features = X.shape[1]
    particles = np.random.randint(2, size=(pop_size, n_features))
    velocities = np.random.uniform(-1, 1, size=(pop_size, n_features))
    
    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(X, y, p) for p in particles])
    gbest = pbest[np.argmax(pbest_fitness)].copy()
    gbest_fitness = np.max(pbest_fitness)
    
    w, c1, c2 = 0.6, 2.0, 2.0
    for iter in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            sigmoid_v = 1 / (1 + np.exp(-velocities[i]))
            particles[i] = (np.random.rand(n_features) < sigmoid_v).astype(int)
            
            fit = fitness_function(X, y, particles[i])
            if fit > pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = fit
                if fit > gbest_fitness:
                    gbest = particles[i].copy()
                    gbest_fitness = fit
                    
    selected_idx = [i for i, bit in enumerate(gbest) if bit == 1]
    if len(selected_idx) == 0: selected_idx = [np.random.randint(n_features)]
    return selected_idx

# 3. SOM-GA (Self-Organizing Map + Genetic Algorithm)
def som_fitness_function(X, y, feature_subset):
    if sum(feature_subset) == 0:
        return 0
    selected_idx = [i for i, bit in enumerate(feature_subset) if bit == 1]
    X_sub = X[:, selected_idx]
    
    som_size = max(2, int(np.sqrt(5 * np.sqrt(X_sub.shape[0]))))
    som = MiniSom(som_size, som_size, X_sub.shape[1], sigma=1.0, learning_rate=0.5)
    som.train_random(X_sub, 50)
    
    q_error = som.quantization_error(X_sub)
    knn_acc = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_sub, y, cv=3).mean()
    
    # Maximize KNN accuracy while penalizing large quantization error 
    # This hybrid fitness pushes SOM-GA to find robust discriminative features
    return knn_acc - (0.01 * q_error)

def som_ga_feature_selection(X, y, pop_size=15, max_iter=25):
    n_features = X.shape[1]
    population = np.random.randint(2, size=(pop_size, n_features))
    
    for iter in range(max_iter):
        fitness = np.array([som_fitness_function(X, y, ind) for ind in population])
        # Ensure positive probabilities
        min_fit = fitness.min()
        max_fit = fitness.max()
        if max_fit == min_fit:
            prob = np.ones(pop_size) / pop_size
        else:
            prob = (fitness - min_fit) / (max_fit - min_fit + 1e-6)
            prob /= prob.sum()
            
        selected_indices = np.random.choice(pop_size, size=pop_size, p=prob, replace=True)
        parents = population[selected_indices]
        
        next_gen = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[min(i+1, pop_size-1)]
            
            crossover_pt = np.random.randint(1, max(2, n_features-1))
            c1 = np.concatenate([p1[:crossover_pt], p2[crossover_pt:]])
            c2 = np.concatenate([p2[:crossover_pt], p1[crossover_pt:]])
            
            for child in [c1, c2]:
                if np.random.rand() < 0.2: # 20% mutation for better search
                    mut_pt = np.random.randint(n_features)
                    child[mut_pt] = 1 - child[mut_pt]
                next_gen.append(child)
                
        population = np.array(next_gen[:pop_size])
        
    final_fitness = np.array([som_fitness_function(X, y, ind) for ind in population])
    best_solution = population[np.argmax(final_fitness)]
    
    selected_idx = [i for i, bit in enumerate(best_solution) if bit == 1]
    if len(selected_idx) == 0: selected_idx = [np.random.randint(n_features)]
    return selected_idx
