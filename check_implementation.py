import numpy as np
from psalg import ParticleSwarm 
import funcs as funcs

def run_pso_multiple_times(num_runs, num_dimensions, lb, ub, function, swarm_size, max_iterations):
    optimal_values = []
    
    for i in range(num_runs):
        pso = ParticleSwarm(function, lb, ub, num_dimensions, options={'SwarmSize': swarm_size, 'MaxIterations': max_iterations})
        _, best_fitness, _ = pso.optimize()
        optimal_values.append(best_fitness)
        print(f"Run {i+1}: Best fitness = {best_fitness}")

    optimal_values = np.array(optimal_values)
    mean_optimal = np.mean(optimal_values)
    std_optimal = np.std(optimal_values)

    return mean_optimal, std_optimal

def run_all_functions(all_functions, num_runs, num_dimensions, lb, ub):
    results = {}
    for func_name in all_functions:
        func = getattr(funcs, func_name)
        print(f"\nRunning PSO on {func_name} function...")
        mean, std = run_pso_multiple_times(num_runs, num_dimensions, lb, ub, func)
        results[func_name] = {"mean": mean, "std": std}
    return results

if __name__ == "__main__":
    # Default values
    default_num_runs = 100
    default_num_dimensions = 3

    # Get user input or use default values
    num_runs = int(input(f"Enter number of runs (default {default_num_runs}): ") or default_num_runs)
    num_dimensions = int(input(f"Enter number of dimensions (default {default_num_dimensions}): ") or default_num_dimensions)
    lb = [-10] * num_dimensions  # Lower bound
    ub = [10] * num_dimensions   # Upper bound
    use_custom_params = input("Do you want to set custom SwarmSize and MaxIterations or would you like to use the default values? (y/n): ").lower() == 'y'

    if use_custom_params:
        swarm_size = int(input("Enter SwarmSize (default is 50): ") or 50)
        max_iterations = int(input("Enter MaxIterations (default is 1000): ") or 1000)
    else:
        swarm_size = 50  # optimal value
        max_iterations = 1000  # optimal value


    print(f"Number of runs: {num_runs}")
    print(f"Number of dimensions: {num_dimensions}")
    print(f"Bounds: [{lb[0]}, {ub[0]}]")


    # Get all functions from funcs module
    all_functions = [func for func in dir(funcs) if callable(getattr(funcs, func)) and not func.startswith("__")]

    while True:
        print("\nAvailable options:")
        print("1. Run PSO on a single function")
        print("2. Run PSO on all functions")
        print("0. Quit")

        choice = input("\nEnter your choice: ")
        
        if choice == '0':
            print("Exiting program.")
            break
        
        elif choice == '1':
            while True:
                print("\nAvailable benchmark functions:")
                for i, func_name in enumerate(all_functions, 1):
                    print(f"{i}. {func_name}")
                print("0. Back to main menu")

                func_choice = input("\nChoose a function to run PSO on (enter number) or enter 0 to go back to main menu: ")
                
                if func_choice == '0':
                    break
                
                try:
                    func_index = int(func_choice) - 1
                    if 0 <= func_index < len(all_functions):
                        func_name = all_functions[func_index]
                        func = getattr(funcs, func_name)
                        
                        print(f"\nRunning PSO on {func_name} function...")
                        mean, std = run_pso_multiple_times(num_runs, num_dimensions, lb, ub, func, swarm_size, max_iterations)
                        
                        print("\nResults:")
                        print(f"  Mean optimal value: {mean}")
                        print(f"  Standard deviation of optimal values: {std}")
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        elif choice == '2':
            print("\nRunning PSO on all functions...")
            results = run_all_functions(all_functions, num_runs, num_dimensions, lb, ub)
            
            print("\nFinal Results:")
            for func_name, result in results.items():
                print(f"{func_name}:")
                print(f"  Mean optimal value: {result['mean']}")
                print(f"  Standard deviation of optimal values: {result['std']}")
        
        else:
            print("Invalid choice. Please enter 0, 1, or 2.")