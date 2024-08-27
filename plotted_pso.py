import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from psalg import ParticleSwarm
import funcs
def plot_3d_function(ax, objective_func, lb, ub):
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
    ax.set_title(f'3D {objective_func.__name__.capitalize()} Function')

def visualize_pso_3d(objective_func):
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

    plot_3d_function(ax1, objective_func, lb, ub)
    plot_3d_function(ax2, objective_func, lb, ub)

    default_num_particles = 100  # Default number of particles
    # Ask user for number of particles
    num_particles = input(f"Enter number of particles (default is {default_num_particles}): ")
    num_particles = int(num_particles) if num_particles.isdigit() else default_num_particles

    default_max_iterations = 500
    max_iterations = input(f"Enter maximum number of iterations performed by PSO (default is {default_max_iterations}): ")
    max_iterations = int(max_iterations) if max_iterations.isdigit() else default_max_iterations

    pso = ParticleSwarm(objective_func, lb, ub, num_dimensions, options={'SwarmSize': num_particles, 'MaxIterations': max_iterations})
    # Plot initial positions
    initial_positions = np.array([p.position for p in pso.particles])
    initial_z = np.array([objective_func(p.position) for p in pso.particles])
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_z, 
                color='red', s=50, label='Initial positions')
    ax1.legend()

    # Optimize
    best_position, best_fitness, _ = pso.optimize()

    # Plot final positions
    final_positions = np.array([p.position for p in pso.particles])
    final_z = np.array([objective_func(p.position) for p in pso.particles])
    ax2.scatter(final_positions[:, 0], final_positions[:, 1], final_z, 
                color='green', s=50, label='Final positions')
    best_z = objective_func(best_position)
    ax2.scatter(best_position[0], best_position[1], best_z, 
                color='red', s=50, label='Best position')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    available_functions = [func for func in dir(funcs) if callable(getattr(funcs, func)) and not func.startswith("__")]
    
    print("Available functions:")
    for i, func in enumerate(available_functions, 1):
        print(f"{i}. {func}")
    
    while True:
        choice = input("Choose a function to visualize (enter number or '0' to exit): ")
        
        if choice == '0':
            print("Exiting...")
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_functions):
                func_name = available_functions[index]
                objective_func = getattr(funcs, func_name)
                print(f"Visualizing {func_name} function...")
                visualize_pso_3d(objective_func)
            else:
                print("Invalid input. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


