import numpy as np
from psalg import ParticleSwarm 
import funcs as funcs

# Default values
default_num_runs = 10
default_num_dimensions = 2
default_num_particles = 50
default_num_iterations = 600
default_tolerance = 1e-6
_verbose = False
minimized = True

def run_pso_multiple_times(num_runs=default_num_runs, num_dimensions=default_num_dimensions, lb=[-10] * default_num_dimensions, ub=[10] * default_num_dimensions, function=funcs.sphere, swarm_size = default_num_particles, max_iterations = default_num_iterations, verbose = False,minimized = True, tolerance = default_tolerance):
    optimal_values = []
    elapsed_times = []

    for i in range(num_runs):
        if verbose:
            print(f"___________________Run [{i+1}]:_______________________{num_runs-i} runs left_")
        try:
            pso = ParticleSwarm(function, lb, ub, num_dimensions, options={'SwarmSize': swarm_size, 'MaxIterations': max_iterations, 'Tolerance': tolerance},minimize = minimized)
            _, best_fitness, elapsed_time , iterations_performed = pso.optimize(verbose)
            optimal_values.append(best_fitness)
            elapsed_times.append(elapsed_time)
            if verbose:
                print(f"Best fitness = {best_fitness}")
        except Exception as e:
            print(f"------------------Error: {str(e)}--------------------")
            exit()
    try:    
        optimal_values = np.array(optimal_values)
        elapsed_times = np.array(elapsed_times)
        mean_optimal = np.mean(optimal_values)
        std_optimal = np.std(optimal_values)
        time_avg = np.mean(elapsed_times)
        return mean_optimal, std_optimal, time_avg
    except Exception as e:
        print(f"------------------Error: {str(e)}--------------------")
        exit()

def print_list(my_list):
    for item in my_list:
        print(item)

def run_all_functions(all_functions, num_runs=default_num_runs, num_dimensions=default_num_dimensions, lb=[-10] * default_num_dimensions, ub=[10] * default_num_dimensions, swarm_size = default_num_particles, max_iterations = default_num_iterations, verbose = False,minimized = True):
    results = {}
    for func_name in all_functions:
        func = getattr(funcs, func_name)
        print(f"\nRunning PSO on {func_name} function...")
        mean, std , time_avg= run_pso_multiple_times(num_runs, num_dimensions, lb, ub, func, swarm_size, max_iterations, verbose,minimized=minimized)
        results[func_name] = {"mean": mean, "std": std, "time_avg": time_avg}
    return results

if __name__ == "__main__":

    # Get user input or use default values
    num_runs = int(input(f"Enter number of runs (default {default_num_runs}): ") or default_num_runs)
    num_dimensions = int(input(f"Enter number of dimensions (default {default_num_dimensions}): ") or default_num_dimensions)
    lb = [-10] * num_dimensions  # Lower bound
    ub = [10] * num_dimensions   # Upper bound
    print("_______________\nDefault values:\n_______________")
    print(f"Swarmsize: {default_num_particles}")
    print(f"Number of iterations: {default_num_iterations}")
    print(f"Tolerance: {default_tolerance}")
    use_default_params = input("Do you want to use the default values for SwarmSize, MaxIterations and Tolerance or set custom values? ([y] = default/n = custom): ").lower()
    if use_default_params == 'y' or use_default_params == "":
        swarm_size = default_num_particles  # optimal value
        max_iterations = default_num_iterations  # optimal value
        tolerance = default_tolerance
    else:
        swarm_size_input = input("Enter SwarmSize (default is 50): ")
        swarm_size = default_num_particles if swarm_size_input == "" else int(swarm_size_input)
        max_iterations_input = input(f"Enter MaxIterations (default is {default_num_iterations}): ")
        max_iterations = default_num_iterations if max_iterations_input == "" else int(max_iterations_input)
        tolerance_input = input(f"Enter Tolerance (default is {default_tolerance}): ")
        tolerance = default_tolerance if tolerance_input == "" else float(tolerance_input)
    minimized_input = input("Do you want to minimize the function? ([y]= minimize/n = maximize): ").lower()
    if minimized_input == 'y' or minimized_input == "":
        minimized = True
    else:
        minimized = False
    del minimized_input
    del use_default_params  
    print("-----------------------------------------SUMMARY-----------------------------------------")
    print(f"Number of runs: {num_runs}")
    print(f"Number of dimensions: {num_dimensions}")
    print(f"Number of particles: {swarm_size}")
    print(f"Number of iterations: {max_iterations}")
    print(f"Tolerance: {tolerance}")
    print(f"Optimization type: {'minimization' if minimized else 'maximization'}")
    print(f"Bounds: [{lb[0]}, {ub[0]}]")
    print(f"Tolerance: {tolerance}")
    # Get all functions from funcs module
    all_functions = [func for func in dir(funcs) if callable(getattr(funcs, func)) and not func.startswith("_") and not func.startswith("parse_user_function")]
    list_of_functions = []
    list_of_options_1 = ["Available options:", "1. Run PSO on a single function", "2. Run PSO on all functions", "0. Quit"]
    list_of_options_2 = ["What would you like to do next?\n", "1. Available benchmark functions:", "0. Exit"]
    list_of_functions.append("Available benchmark functions:\n")  
    for i, func_name in enumerate(all_functions, 1):
                    list_of_functions.append(str(i)+". "+str(func_name))
    list_of_functions.append("0. Back to main menu")

    print("-----------------------------------------------------------------------------------------")
    while True:
        print_list(list_of_options_1)
        choice = input("\nEnter your choice: ")
        if choice == '0':
            print("Exiting program.")
            break
        elif choice == '1':
            v = input("Do you want to see PSO in action? ([y] = yes/n = no): ").lower()
            if  v == 'y':
                _verbose = True
            while choice == '1':
                print_list(list_of_functions)
                func_choice = input("\nChoose a function to run PSO on (enter number) or enter 0 to go back to main menu: ")
                if func_choice == '0':
                    break
                try:
                    func_index = int(func_choice) - 1
                    if 0 <= func_index < len(all_functions):
                        func_name = all_functions[func_index]
                        func = getattr(funcs, func_name)
                        print("--------------------------------------------")
                        print(f"Running PSO on {func_name} function...")
                        print("--------------------------------------------")
                        mean, std, avg_time = run_pso_multiple_times(num_runs, num_dimensions, lb, ub, func, swarm_size, max_iterations, _verbose)
                        print("_________\nResults:\n_________")
                        print(f"  Mean optimal value: {mean}")
                        print(f"  Standard deviation of optimal values: {std}")
                        print(f"  Average time per run: {avg_time: 0.4f} seconds")
                        print("----------------------------------------------------------------------------------------------")
                        while True:
                            print_list(list_of_options_2)
                            choice = input("Enter your choice: ")
                            if choice == '1':
                                break
                            elif choice == '0':
                                print("Exiting program.")
                                exit()
                            else:
                                print("Invalid choice. Please enter 0 or 1.")
                        if choice == '1':
                            break  # Break the inner while loop to go back to the main menu
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.") 
        elif choice == '2':
            v = input("Do you want to see PSO in action? ([y] = yes/n = no): ").lower()
            if  v == 'y':
                _verbose = True
            print("\nRunning PSO on all functions...")
            results = run_all_functions(all_functions, num_runs, num_dimensions, lb, ub, swarm_size, max_iterations, _verbose, minimized=minimized)
            print("-----------------------------------------FINAL_RESULTS------------------------------------------------")
            print("\nFinal Results:")
            
            # Create and write to the summary file
            summary_file = 'pso_results_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"PSO Results Summary\n")
                f.write(f"Number of runs: {num_runs}\n")
                f.write(f"Number of dimensions: {num_dimensions}\n")
                f.write(f"Number of particles: {swarm_size}\n")
                f.write(f"Number of iterations: {max_iterations}\n")
                f.write(f"Tolerance: {tolerance}\n")
                f.write(f"Optimization type: {'minimization' if minimized else 'maximization'}\n")
                f.write(f"Bounds: [{lb[0]}, {ub[0]}]\n\n")
                f.write("Function Name | Mean Optimal Value | Standard Deviation | Average Time (s)\n")
                f.write("-" * 75 + "\n")
            
            for func_name, result in results.items():
                print(f"{func_name}:")
                print(f"  Mean optimal value: {result['mean']}")
                print(f"  Standard deviation of optimal values: {result['std']}")
                print(f"  Average time per run: {result['time_avg']:0.4f} seconds")
                print("----------------------------------------------------------------------------------------")
                
                # Append to the summary file
                with open(summary_file, 'a') as f:
                    f.write(f"{func_name:<14} | {result['mean']:<18.6f} | {result['std']:<19.6f} | {result['time_avg']:0.4f}\n")
            
            print(f"\nResults summary has been saved to '{summary_file}'")
            
            # Open the file after creation
            import os
            import subprocess
            
            try:
                if os.name == 'nt':  # For Windows
                    os.startfile(summary_file)
                elif os.name == 'posix':  # For macOS and Linux
                    opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                    subprocess.call([opener, summary_file])
                print(f"The file '{summary_file}' has been opened.")
            except Exception as e:
                print(f"An error occurred while trying to open the file: {e}")
                print(f"Please open '{summary_file}' manually to view the results.")

        else:
            print("Invalid choice. Please enter 0, 1, or 2.")