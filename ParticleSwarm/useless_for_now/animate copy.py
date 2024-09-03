import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
from psalg import ParticleSwarm

def parse_user_function(func_str):
    """Parse user input string into a callable function."""
    try:
        return lambda x: eval(func_str, {"x": x, "np": np})
    except:
        print("Invalid function. Please try again.")
        return None

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

    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    plot_3d_function(ax_anim, objective_func, lb, ub, is_user_defined)

    # Initialize empty lists for scatter and quiver plots
    scatter = None
    quiver = None

    default_num_particles = 100  # Default number of particles
    # Ask user for number of particles
    num_particles = input(f"Enter number of particles (default is {default_num_particles}): ")
    num_particles = int(num_particles) if num_particles.isdigit() else default_num_particles

    default_max_iterations = 500
    max_iterations = input(f"Enter maximum number of iterations performed by PSO (default is {default_max_iterations}): ")
    max_iterations = int(max_iterations) if max_iterations.isdigit() else default_max_iterations


    minimize = input("Minimize the function? (y/n): ").lower() == 'y'
    

    pso = ParticleSwarm(objective_func, lb, ub, num_dimensions, options={'SwarmSize': num_particles, 'MaxIterations': max_iterations}, minimize=minimize)
    # Plot initial positions
    initial_positions = np.array([p.position for p in pso.particles])
    initial_z = np.array([objective_func(p.position) for p in pso.particles])
    ax_anim.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_z, 
                    color='magenta', s=35, label='Initial positions')
    ax_anim.legend()

    # Initialize empty lists for scatter and quiver plots
    scatter = None
    quiver = None

    for iteration in range(max_iterations):
        # Get current positions and evaluate
        current_positions = np.array([p.position for p in pso.particles])
        current_z = np.array([objective_func(p.position) for p in pso.particles])

        # Remove previous scatter and quiver plots
        if scatter:
            scatter.remove()
        if quiver:
            quiver.remove()

        # Plot current positions
        scatter = ax_anim.scatter(current_positions[:, 0], current_positions[:, 1], current_z, 
                                  color='orange', s=35, label='Current positions')

        # Perform one iteration of PSO
        pso._update_particles(iteration)
        for particle in pso.particles:
            particle.evaluate(objective_func, pso.minimize)

        # Get new positions after update
        new_positions = np.array([p.position for p in pso.particles])
        new_z = np.array([objective_func(p.position) for p in pso.particles])

        # Plot directions using quiver
        quiver = ax_anim.quiver(current_positions[:, 0], current_positions[:, 1], current_z,
                                new_positions[:, 0] - current_positions[:, 0],
                                new_positions[:, 1] - current_positions[:, 1],
                                new_z - current_z,
                                color='red', length=1, normalize=True)

        # Update title with iteration number
        ax_anim.set_title(f'PSO Iteration {iteration + 1}')

        # Pause to create animation effect
        plt.pause(0.1)

    # Plot final positions and best position
    final_positions = np.array([p.position for p in pso.particles])
    final_z = np.array([objective_func(p.position) for p in pso.particles])
    ax_anim.scatter(final_positions[:, 0], final_positions[:, 1], final_z, 
                    color='green', s=35, label='Final positions')
    
    best_position, best_fitness, _ = pso.optimize()
    best_z = objective_func(best_position)
    ax_anim.scatter(best_position[0], best_position[1], best_z, 
                    color='blue', s=100, label='Best position')
    ax_anim.legend()

    plt.show()

    # ... (keep the rest of the function as is)
    
def enumerate_functions(filename):
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
    available_functions = [func for func in dir(module) if callable(getattr(module, func)) and not func.startswith("__")]

    print("Available functions:")
    for i, func in enumerate(available_functions, 1):
        print(f"{i}. {func}")
    print(f"{len(available_functions) + 1}. Input custom function")
    return available_functions

# Update the available_functions in the main block

if __name__ == "__main__":
    available_functions = enumerate_functions("funcs")
    
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
                objective_func = getattr(importlib.import_module("funcs"), func_name)
                print(f"Visualizing {func_name} function...")
                visualize_pso_3d(objective_func)
            elif index == len(available_functions):
                print("Enter your custom function using 'x' as the input variable.")
                print("Example: np.sin(x[0]) + np.cos(x[1])")
                user_func_str = input("Function: ")
                user_func = parse_user_function(user_func_str)
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
