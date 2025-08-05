# src/feature_selection.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import random

def logistic_map(size, x0=0.7, r=3.9):
    """
    Generate chaotic sequence using logistic map
    """
    seq = np.zeros(size)
    seq[0] = x0
    for i in range(1, size):
        seq[i] = r * seq[i-1] * (1 - seq[i-1])
    return seq

def fitness_function(X, y, feature_subset):
    """
    Evaluate accuracy of classifier on selected features
    """
    if sum(feature_subset) == 0:
        return 0  # no features selected
    selected_idx = [i for i, bit in enumerate(feature_subset) if bit == 1]
    clf = KNeighborsClassifier(n_neighbors=3)
    score = cross_val_score(clf, X[:, selected_idx], y, cv=3).mean()
    return score

def reptile_search_algorithm(X, y, pop_size=10, max_iter=20):
    """
    Reptile Search Algorithm with chaotic logistic map initialization
    """
    n_features = X.shape[1]

    # Chaotic logistic map to initialize population (binary masks)
    chaotic_seq = logistic_map(pop_size * n_features)
    chaotic_seq = chaotic_seq.reshape((pop_size, n_features))
    population = (chaotic_seq > 0.5).astype(int)

    # Evaluate fitness
    fitness = [fitness_function(X, y, ind) for ind in population]

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    print(f"Initial best fitness: {best_fitness:.4f}")

    for iter in range(max_iter):
        for i in range(pop_size):
            # Reptile position update: simplified random swap based on best
            new_individual = population[i].copy()
            for j in range(n_features):
                if random.random() < 0.3:  # exploration
                    new_individual[j] = 1 - new_individual[j]
                elif random.random() < 0.2:  # exploitation
                    new_individual[j] = best_solution[j]
            new_fit = fitness_function(X, y, new_individual)
            # Greedy update
            if new_fit > fitness[i]:
                population[i] = new_individual
                fitness[i] = new_fit
                # Update global best
                if new_fit > best_fitness:
                    best_solution = new_individual.copy()
                    best_fitness = new_fit
        print(f"Iteration {iter+1}/{max_iter}, Best fitness: {best_fitness:.4f}")

    selected_idx = [i for i, bit in enumerate(best_solution) if bit == 1]
    print("Selected feature indices:", selected_idx)
    return selected_idx

if __name__ == "__main__":
    # Example usage after feature engineering
    from data_loader import load_pv_data
    from feature_engineering import create_features

    file_path = "../data/pv_data_sample.xlsx"
    df = load_pv_data(file_path)
    df_fe = create_features(df)

    # Input features (drop outputs)
    output_cols = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc']
    input_cols = [col for col in df_fe.columns if col not in output_cols + ['Condition_ID', 'Condition_Name', 'Row', 'Col']]

    X = df_fe[input_cols].to_numpy()
    # For classification: create label, e.g., Condition_ID
    y = df_fe['Condition_ID'].to_numpy()

    selected = reptile_search_algorithm(X, y)
    print("\nSelected feature names:")
    print([input_cols[i] for i in selected])




# real Reptile Search Algorithm with intialization , exploration and exploitation, fitness evaluation using classifer

# # src/feature_selection.py
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier

# def logistic_map_sequence(size, x0=0.7, r=3.9):
#     """
#     Generate chaotic sequence with logistic map.
#     """
#     seq = np.zeros(size)
#     seq[0] = x0
#     for i in range(1, size):
#         seq[i] = r * seq[i - 1] * (1 - seq[i - 1])
#     return seq

# def transfer_function(position):
#     """
#     Transfer function to map continuous position to binary:
#     Use sigmoid.
#     """
#     return 1 / (1 + np.exp(-10 * (position - 0.5)))  # steep sigmoid

# def binary_solution(position):
#     """
#     Apply transfer and get binary mask.
#     """
#     prob = transfer_function(position)
#     return (np.random.rand(*prob.shape) < prob).astype(int)

# def fitness_function(X, y, feature_subset):
#     """
#     Evaluate classifier accuracy on selected features.
#     """
#     if np.sum(feature_subset) == 0:
#         return 0
#     selected_idx = np.where(feature_subset == 1)[0]
#     clf = KNeighborsClassifier(n_neighbors=3)
#     score = cross_val_score(clf, X[:, selected_idx], y, cv=3).mean()
#     return score

# def reptile_search_algorithm(X, y, pop_size=10, max_iter=30):
#     """
#     More realistic RSA with chaotic map, transfer function, control params.
#     """
#     n_features = X.shape[1]

#     # Initialize chaotic map for each individual
#     chaotic = logistic_map_sequence(pop_size)

#     # Initialize population: each is position vector in [0,1]
#     population = np.random.rand(pop_size, n_features)

#     # Evaluate initial fitness
#     binary_pop = np.array([binary_solution(ind) for ind in population])
#     fitness = np.array([fitness_function(X, y, ind) for ind in binary_pop])

#     best_idx = np.argmax(fitness)
#     best_pos = population[best_idx].copy()
#     best_fit = fitness[best_idx]

#     print(f"Initial best fitness: {best_fit:.4f}")

#     for iter in range(max_iter):
#         α = 2 * (1 - iter / max_iter)  # linearly decreasing parameter

#         for i in range(pop_size):
#             r1 = np.random.rand(n_features)
#             r2 = np.random.rand(n_features)

#             # Update chaotic number
#             chaotic[i] = 3.9 * chaotic[i] * (1 - chaotic[i])

#             # RSA-inspired position update (simplified)
#             new_pos = population[i] + α * r1 * (best_pos - population[i]) + chaotic[i] * r2 * (np.random.rand(n_features) - 0.5)

#             # Bound to [0,1]
#             new_pos = np.clip(new_pos, 0, 1)

#             # Evaluate new
#             new_bin = binary_solution(new_pos)
#             new_fit = fitness_function(X, y, new_bin)

#             # Greedy update
#             if new_fit > fitness[i]:
#                 population[i] = new_pos
#                 fitness[i] = new_fit

#                 # Update global best
#                 if new_fit > best_fit:
#                     best_fit = new_fit
#                     best_pos = new_pos.copy()

#         if (iter + 1) % 5 == 0 or iter == max_iter - 1:
#             print(f"Iteration {iter+1}/{max_iter}, Best fitness: {best_fit:.4f}")

#     # Final selected features
#     best_binary = binary_solution(best_pos)
#     selected_idx = np.where(best_binary == 1)[0]
#     print("Selected feature indices:", selected_idx)
#     return selected_idx

# if __name__ == "__main__":
#     from data_loader import load_pv_data
#     from feature_engineering import create_features

#     file_path = "../data/pv_data_sample.xlsx"
#     df = load_pv_data(file_path)
#     df_fe = create_features(df)

#     # Input features: drop outputs & meta columns
#     output_cols = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc']
#     input_cols = [col for col in df_fe.columns if col not in output_cols + ['Condition_ID', 'Condition_Name', 'Row', 'Col']]

#     X = df_fe[input_cols].to_numpy()
#     y = df_fe['Condition_ID'].to_numpy()

#     selected = reptile_search_algorithm(X, y)

#     print("\nSelected feature names:")
#     print([input_cols[i] for i in selected])
