import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
from psalg import ParticleSwarm
from funcs import parse_user_function
from plotted_pso import enumerate_functions
from drw import create_gif_from_images


def start_anim(objective_func, is_user_defined = False):
    path = 'ParticleSwarm/pso_figures'
    num_dimensions = 2
    num_particles = 30
    tolerance = 0.01
    lb = [-5.12, -5.12]
    ub = [5.12, 5.12]
    max_iterations = 50
    isMinimized=True
    setting = int(input("   1. Minimize Function\n   2. Maximize Function\n"))
    
    if setting == 1:
        isMinimized = True
    elif setting == 2:
        isMinimized = False
    else:
        print("Invalid choice, using minimization instead")
    optimization_type = "minimization" if isMinimized else "maximization"
    setting = int(input("   1. Custom Max Iterations\n   2. Default Max Iterations [50]\n"))
    if setting == 1:
        max_iterations = int(input("Enter number of max iteration: "))
    elif setting == 2:
        pass
    else:
        print("Invalid choice, using default value instead")
    setting = int(input("   1. Custom Number of Particles\n   2. Default Number of Particles [30]\n"))
    if setting == 1:
        num_particles = int(input("Enter number of particles: "))
    elif setting == 2:
        pass
    else:
        print("Invalid choice, using default value instead")
    del setting
    pso = ParticleSwarm(objective_func, lb, ub, num_dimensions, options={'SwarmSize': num_particles, 'MaxIterations': max_iterations, 'Tolerance':tolerance}, minimize= isMinimized, isanimated=True, is_user_defined=is_user_defined)
    pso.optimize(path = path)
    print("Creating your GIF")
    create_gif_from_images(objective_func=objective_func, optimization_type=optimization_type, folder_path  = path, is_user_defined=is_user_defined)

def main():
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
                objective_func = getattr(importlib.import_module("funcs"), func_name)
                print(f"Visualizing {func_name} function...")
                start_anim(objective_func)
            elif index == len(available_functions):
                user_func = parse_user_function()
                if user_func:
                    start_anim(user_func, is_user_defined = True)
            else:
                print("Invalid input. Please choose a number from the list.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        # Add prompt to visualize another function or exit
        visualize_again = int(input("   1. Visualize another function\n   0. Exit\n"))
        if visualize_again != 1:
            print("Exiting...")
            break
