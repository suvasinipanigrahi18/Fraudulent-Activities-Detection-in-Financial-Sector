import time
import numpy as np


def gauss_mutation(mu, sigma):
    # Generate a random value using Gaussian mutation
    return np.random.normal(loc=mu, scale=sigma)

#  Improved Waterwheel Plant Algorithm (IWPA)

def PROPOSED(positions, obj_fun, ub,lb ,T_max):
    # Initialize plant positions
    n = positions.shape[0], positions.shape[1]

    # Initialize best position and best fitness
    best_position = np.zeros((1, n))
    new_best_fitness = obj_fun(positions)
    r1 = 0.1
    r2 = 0.2
    r3 = 0.3
    f = 0.5
    c = 0.1
    k = 1.0

    # Initialize variables
    t = 1
    K_value = k
    Convergence_curve = np.zeros(T_max)
    ct = time.time()
    r = (np.min(new_best_fitness) ^ 1.5) / np.max(new_best_fitness)
    # Main optimization loop
    while t <= T_max:
        for i in range(n):
            if r < 0.5:
                # Explore using W→
                W = r1 * (best_position + 2 * K_value * positions[i])
                new_position = positions[i] + W * (2 * K_value + r2)

                if obj_fun(new_position) == obj_fun(positions[i]):
                    positions[i] = gauss_mutation(np.mean(positions), np.std(positions)) + r1 * (
                                positions[i] + 2 * K_value * W)
            else:
                # Exploit using W→
                W = r3 * (k * best_position[i] + r3 * positions[i])
                new_position = positions[i] + K_value * W

                if obj_fun(new_position) == obj_fun(positions[i]):
                    positions[i] = (r1 + K_value) * np.sin(f * c * np.pi)

        # Decrease K exponentially
        K_value = (1 + 2 * t ** 2 / (T_max ** 3 + f))

        positions = np.clip(positions, lb, ub)

        # Update best position and best fitness
        current_best_fitness = obj_fun(best_position)
        new_best_fitness = obj_fun(positions)
        if new_best_fitness < current_best_fitness:
            best_position = positions.copy()

        t += 1
        Convergence_curve[t] = new_best_fitness
    ct = time.time() - ct

    return new_best_fitness, Convergence_curve, best_position, ct

