import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

def plot_3d_function(ax, objective_func, lb, ub, is_user_defined=False):
    x = np.linspace(lb[0], ub[0], 100)
    y = np.linspace(lb[1], ub[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if is_user_defined:
        ax.set_title("User-defined Function")
    else:
        ax.set_title(f'3D {objective_func.__name__.capitalize()} Function')
        

def visualize_pso_3d(objective_func, is_user_defined=False):
    from psalg import ParticleSwarm
    num_dimensions = 2
    # Default bounds
    default_lb = [-5.12, -5.12]
    default_ub = [5.12, 5.12]

    # Ask user if they want to specify custom bounds
    use_custom_bounds = input("Do you want to specify custom bounds? (y/n): ").lower() == 'y'

    if use_custom_bounds:
        lb = [float(x) for x in input("Enter lower bounds (comma-separated): ").split(',')]
        ub = [float(x) for x in input("Enter upper bounds (comma-separated): ").split(',')]
    else:
        lb = default_lb
        ub = default_ub

    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    plot_3d_function(ax1, objective_func, lb, ub, is_user_defined)
    plot_3d_function(ax2, objective_func, lb, ub, is_user_defined)

    default_num_particles = 50  
    num_particles = input(f"Enter number of particles (default is {default_num_particles}): ")
    num_particles = int(num_particles) if num_particles.isdigit() else default_num_particles

    default_max_iterations = 500
    max_iterations = input(f"Enter maximum number of iterations performed by PSO (default is {default_max_iterations}): ")
    max_iterations = int(max_iterations) if max_iterations.isdigit() else default_max_iterations
    minimize = input("Minimize the function? (y/n): ").lower() == 'y'
    pso = ParticleSwarm(objective_func, lb, ub, num_dimensions, options={'SwarmSize': num_particles, 'MaxIterations': max_iterations}, minimize= minimize)

    # Plot initial positions
    initial_positions = np.array([p.position for p in pso.particles])
    initial_z = np.array([p.evaluate(objective_func, minimize) for p in pso.particles])
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_z, color='magenta', s=35, label='Initial positions')
    ax1.legend()
    best_position, best_fitness, _ , _ = pso.optimize(verbose=True)
    final_positions = np.array([p.position for p in pso.particles])
    final_z = np.array([p.evaluate(objective_func, minimize) for p in pso.particles])

    ax2.scatter(final_positions[:, 0], final_positions[:, 1], final_z, 
                color='orange', s=35, label='Final positions')
    best_z = objective_func(best_position)
    ax2.scatter(best_position[0], best_position[1], best_z, 
                color='blue', s=100, label='Best position')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
def enumerate_functions(filename, _verbose = True):
    """
    Enumerate all functions in the specified module.
    
    Args:
    filename (str): The name of the local Python file to import (without .py extension).
    
    Returns:
    list: A list of function names available for optimization.
    """    
    # Dynamically import the module
    module = importlib.import_module(filename) 
    # Get all functions from the imported module
    available_functions = [func for func in dir(module) if callable(getattr(module, func)) and not func.startswith("parse_user_function")]
    if _verbose:
        print("Available functions:")
        for i, func in enumerate(available_functions, 1):
            print(f"{i}. {func}")
        print(f"{len(available_functions) + 1}. Input custom function")
    return available_functions

# Update the available_functions in the main block
if __name__ == "__main__":
    available_functions = enumerate_functions("funcs", _verbose = False)
    while True:
        print("\nAvailable functions:")
        for i, func in enumerate(available_functions, 1):
            print(f"{i}. {func}")
        print(f"{len(available_functions) + 1}. Input custom function")
        choice = input("Choose a function to visualize (enter number or '0' to exit): ")
        if choice == '0':
            print("Exiting...")
            break
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_functions):
                func_name = available_functions[index]
                # This line dynamically imports the chosen function from the 'funcs' module
                # It first imports the 'funcs' module, then uses getattr to get the function with the name stored in func_name from that module
                objective_func = getattr(importlib.import_module("funcs"), func_name)
                print(f"Visualizing {func_name} function...")
                visualize_pso_3d(objective_func)
            elif index == len(available_functions):
                user_function = getattr(importlib.import_module("funcs"), "parse_user_function")
                user_func = user_function()
                if user_func:
                    print("Visualizing user-defined function...")
                    visualize_pso_3d(user_func, is_user_defined=True)
            else:
                print("Invalid input. Please choose a number from the list.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        # Prompt user after visualization
        continue_choice = input("\nWould you like to visualize another function? (y/n): ").lower()
        if continue_choice != 'y':
            print("Exiting...")
            break