import numpy as np
from psalg import ParticleSwarm 
from funcs import sphere, rastrigin

def run_pso_multiple_times(num_runs, num_dimensions, lb, ub, function):
    optimal_values = []

    for i in range(num_runs):
        pso = ParticleSwarm(function, lb, ub, num_dimensions)
        _, best_fitness, _ = pso.optimize()
        optimal_values.append(best_fitness)
        print(f"Run {i+1}: Best fitness = {best_fitness}")

    optimal_values = np.array(optimal_values)
    mean_optimal = np.mean(optimal_values)
    std_optimal = np.std(optimal_values)

    return mean_optimal, std_optimal

if __name__ == "__main__":
    # Default values
    default_num_runs = 100
    default_num_dimensions = 3

    # Get user input or use default values
    num_runs = int(input(f"Enter number of runs (default {default_num_runs}): ") or default_num_runs)
    num_dimensions = int(input(f"Enter number of dimensions (default {default_num_dimensions}): ") or default_num_dimensions)
    lb = [-10] * num_dimensions  # Lower bound
    ub = [10] * num_dimensions   # Upper bound

    print(f"Running PSO on Sphere function {num_runs} times...")
    print(f"Number of dimensions: {num_dimensions}")
    print(f"Bounds: [{lb[0]}, {ub[0]}]")

    mean_sphere, std_sphere = run_pso_multiple_times(num_runs, num_dimensions, lb, ub, sphere)
    mean_rastrigin, std_rastrigin = run_pso_multiple_times(num_runs, num_dimensions, lb, ub, rastrigin)

    print(f"\nResults after {num_runs} runs of sphere function:")
    print(f"Mean optimal value: {mean_sphere}")
    print(f"Standard deviation of optimal values: {std_sphere}")

    print(f"\nResults after {num_runs} runs of rastrigin function:")
    print(f"Mean optimal value: {mean_rastrigin}")
    print(f"Standard deviation of optimal values: {std_rastrigin}")