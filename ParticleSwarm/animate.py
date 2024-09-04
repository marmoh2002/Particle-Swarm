import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
from psalg import ParticleSwarm
from funcs import parse_user_function
from plotted_pso import enumerate_functions
from drw import create_gif_from_images

def start_anim(objective_func):
    path = 'pso_figures'
    num_dimensions = 2
    num_particles = 20
    lb = [-5.12, -5.12]
    ub = [5.12, 5.12]
    max_iterations = 100
    minimize = True
    pso = ParticleSwarm(objective_func, lb, ub, num_dimensions, options={'SwarmSize': num_particles, 'MaxIterations': max_iterations}, minimize= minimize, isanimated=True)
    pso.optimize(path = path)
    print("Good day")
    create_gif_from_images(path)
    
    exit()

if __name__ == "__main__":
    available_functions = enumerate_functions("funcs")
    
    while True:
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
                print("Enter your custom function using 'x' as the input variable.")
                print("Example: np.sin(x[0]) + np.cos(x[1])")
                user_func_str = input("Function: ")
                user_func = parse_user_function(user_func_str)
                if user_func:
                    start_anim(user_func)
            else:
                print("Invalid input. Please choose a number from the list.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

