import os
import matplotlib.pyplot as plt
from plotted_pso import visualize_pso_3d
import funcs

def plot_and_save_function(func, func_name, output_dir):
    print(f"Plotting {func_name}...")
    
    # Create a figure
    plt.figure(figsize=(20, 9))
    
    # Monkey-patch the input function to always return default values
    original_input = __builtins__.input
    __builtins__.input = lambda _: ''
    
    # Call visualize_pso_3d with the patched input function
    visualize_pso_3d(func, is_user_defined=False)
    
    # Restore the original input function
    __builtins__.input = original_input
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"{func_name}_plot.png")
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Saved plot for {func_name} to {plot_filename}")

# Create a directory to save the plots
output_dir = "benchmark_plots"
os.makedirs(output_dir, exist_ok=True)

# Get all benchmark functions from funcs module
benchmark_functions = [func for func in dir(funcs) if callable(getattr(funcs, func)) and not func.startswith("__")]

# Iterate through all benchmark functions
for func_name in benchmark_functions:
    func = getattr(funcs, func_name)
    plot_and_save_function(func, func_name, output_dir)

print("All benchmark functions have been plotted and saved.")