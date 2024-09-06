import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import funcs
import time
from global_minima import get_global_minima
from domain import get_function_domain

def mfo_main():
    # parameters:
    n_moths = 50  
    max_iterations = 1000  
    num_runs = 10
    dim = 2  # Dimensions
    early_stopping_threshold = 1e-6  # Threshold value
    patience = 20  # Number of iterations with no improvement before stopping

    # Get the benchmark functions from the funcs module
    benchmark_functions = get_benchmark_functions()

    # Choose the benchmark function to optimize
    print('Choose a benchmark function:')
    for i, func in enumerate(benchmark_functions, 1):
        print(f'{i}: {func.__name__}')  # Display available benchmark functions
    choice = int(input('Enter the function number: ')) - 1  # User selects a function

    # Select the corresponding benchmark function
    benchmark_func = benchmark_functions[choice]

    # Get the domain for the selected function:
    domain = get_function_domain(benchmark_func.__name__)
    if domain:
        lb, ub = domain
    else:
        print("Domain for the selected function is not available.")
        return

    # Load global minima:
    global_minima = get_global_minima()
    func_name = benchmark_func.__name__
    if func_name in global_minima:
        global_minimum = global_minima[func_name]
    else:
        print("Global minimum for the selected function is not available.")
        return

    # Store results and solutions for all runs
    results = []  
    best_solutions = []  

    # Multiple optimization runs:
    for run in range(num_runs):
        # Initialize moths' positions part:
        moths = lb + (ub - lb) * np.random.rand(n_moths, dim)  # Random
        initial_moths = moths.copy()  # Store initial positions for comparison 
        all_moth_positions = []  # Track positions of all moths at each iteration

        # To store the best solution and position at each run:
        best_solution = np.inf  # Initialize to infinity 
        best_position = np.zeros(dim)  # Initialize to zero vector
        last_improvement_iteration = 0  # Counter for early stopping

        start_time = time.time() # Start time

        for iteration in range(max_iterations):
            # Evaluate the fitness of each moth:
            for i in range(n_moths):
                fitness = benchmark_func(moths[i])
                if np.issubdtype(type(fitness), np.number):  # Check if fitness is a scalar
                    if fitness < best_solution:
                        best_solution = fitness
                        best_position = moths[i].copy()
                else:  #fitness is an array
                    fitness_value = np.min(fitness)
                    if fitness_value < best_solution:
                        best_solution = fitness_value
                        best_position = moths[i].copy()

            all_moth_positions.append(moths.copy())    # Store the positions of moths for animation

            # Check for early stopping:
            if iteration > 0 and abs(prev_best_solution - best_solution) < early_stopping_threshold:
                last_improvement_iteration += 1
                if last_improvement_iteration >= patience:
                    print(f"Early stopping at iteration {iteration + 1} due to lack of improvement.")
                    break
            else:
                last_improvement_iteration = 0

            # Store the best solution from the previous iteration
            prev_best_solution = best_solution

            # Results:
            n_flames = round(n_moths - iteration * (n_moths - 1) / max_iterations)
            moths = update_moths(moths, best_position, lb, ub, n_flames, iteration, max_iterations) # Update the moths' positions
            moths = np.clip(moths, lb, ub) # Clipping to avoid divergence of solutions. 

        end_time = time.time()   # End time
        time_taken = end_time - start_time  # Calculating the time for the run

        # Save the best solution and position found during the run:
        results.append(best_solution)
        best_solutions.append(best_position)

        # Display the outputs:
        print(f'Run {run+1}: Best Solution: {best_solution:.4f}')
        print('Best position:')
        print(best_position)
        print(f'Time taken for this run: {time_taken:.2f} seconds')
        print("_____________________________________________________")

        # Detect if trapped in a local minimum:
        check_local_minimum(best_position, global_minimum)

        # Animate the optimization process to visualize moth movement:
        animate_optimization_paths(all_moth_positions, lb, ub)

    # Calculating the mean and standard deviation for all runs:
    mean_result = np.mean(results)
    std_result = np.std(results)
    # Display the outputs:
    print(f'Mean of best solutions: {mean_result:.4f}')
    print(f'Standard deviation of best solutions: {std_result:.4f}')

    # 3D Plot of the benchmark function and best positions:
    plot_3d_function(benchmark_func, lb, ub, best_solutions,all_moth_positions)

    # Plot the moth positions before and after optimization:
    plot_before_after_optimization(initial_moths, moths, best_solutions)

def update_moths(moths, best_position, lb, ub, n_flames, iteration, max_iterations):
    b = 1  # Constant value
    t = -1 + (iteration / max_iterations) * 2  # Linearly decreases [-1 , 1] 

    # Updating the position of moth part:
    for i in range(moths.shape[0]):
        for j in range(moths.shape[1]):
            distance_to_flame = np.abs(moths[i, j] - best_position[j]) # Distance to the flame calc
            moths[i, j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + best_position[j] # Update position of moth:

    # Ensure moths are within boundaries:
    moths = np.clip(moths, lb, ub)
    return moths

# Benchmark functions part:
def get_benchmark_functions():
    return [getattr(funcs, f) for f in dir(funcs) if callable(getattr(funcs, f)) and not f.startswith("_")]

# 3D part with animation:
def plot_3d_function(func, lb, ub, best_positions, all_moth_positions):
    # Grid to plot the function:
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Function values calc 
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)  # Surface plot of the function
    
    # Scatter for moths' positions (empty initially for animation)
    scat = ax.scatter([], [], [], color='red', marker='o', s=50, label='Moths')

    # Plot the best positions as final points
    best_positions = np.array(best_positions)
    func_values = np.array([func(pos) for pos in best_positions])
    ax.scatter(best_positions[:, 0], best_positions[:, 1], func_values, color='blue', marker='x', s=100, label='Best Solutions')

    ax.set_title(f'3D Plot of {func.__name__}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Objective Function')
    plt.legend()

    # Animation update function
    def update(frame):
        moth_positions = all_moth_positions[frame]
        func_values = np.array([func(pos) for pos in moth_positions])
        scat._offsets3d = (moth_positions[:, 0], moth_positions[:, 1], func_values)
        return scat,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(all_moth_positions), blit=False, repeat=False)

    plt.show()

# Plotting moths before and after optimization and best solutions:
def plot_before_after_optimization(initial_moths, final_moths, best_solutions):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Before optimization:
    ax1.scatter(initial_moths[:, 0], initial_moths[:, 1], c='red')
    ax1.set_title('Moths Before Optimization')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # After optimization:
    ax2.scatter(final_moths[:, 0], final_moths[:, 1], c='green')
    ax2.scatter(np.array(best_solutions)[:, 0], np.array(best_solutions)[:, 1], c='blue', marker='x', s=100, label='Best Solutions')
    ax2.set_title('Moths After Optimization')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.legend()
    plt.show()

# Animation:
def animate_optimization_paths(all_moth_positions, lb, ub):
    fig, ax = plt.subplots()
    
    # Scatter plot for moth positions
    scat = ax.scatter([], [], c='blue') # Empty scatter
    
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.set_title("Moth Optimization Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    def update(frame):
        scat.set_offsets(all_moth_positions[frame]) # Updating the scatter
        return scat,
    
    # The animation part:
    ani = animation.FuncAnimation(fig, update, frames=len(all_moth_positions), blit=True, repeat=False)
    plt.show()

# Checking local min part with very small tolerance:
def check_local_minimum(best_position, global_minimum, tolerance=1e-2):
    if np.allclose(best_position, global_minimum, atol=tolerance):
        print(f"Solution is close to the global minimum (within tolerance of {tolerance})")
    else:
        print(f"Solution is NOT close to the global minimum (possible local minimum)")

if __name__ == "__main__":
    mfo_main()