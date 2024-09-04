import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotted_pso import plot_3d_function
import os
import imageio
import glob
import platform
import subprocess

def draw_frame(particles, objective_func, run_num, path):
    num_dimensions = 2
    # Default bounds
    lb = [-5.12, -5.12]
    ub = [5.12, 5.12]
    minimize = True
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_function(ax, objective_func, lb, ub)
    positions = np.array([p.position for p in particles])
    z = np.array([p.evaluate(objective_func, minimize) for p in particles])
    ax.scatter(positions[:, 0], positions[:, 1], z, color='red', s=45, label='particles')
    ax.legend()
    ax.set_title(f"Run number {run_num}")
    plt.tight_layout()
    # plt.show(block=False)
    print(f"loading gif... {run_num}")
    if run_num == 1:
        if os.path.exists(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(path)
    plt.savefig(f'pso_figures/{run_num}.png')
    plt.close(fig)
       
def create_gif_from_images(objective_func, optimization_type, folder_path, duration=5):
    """
    Create a repeating GIF from a folder of images

    :param folder_path: Path to the folder containing the images
    :param output_filename: Name of the output GIF file
    :param duration: Duration of each frame in the GIF (in seconds)
    """
    # Get list of PNG files in the folder, sorted by name
    images = sorted(glob.glob(f"{folder_path}/*.png"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    output_filename = f"{objective_func.__name__}_{optimization_type}.gif"
    # Read in all the images
    image_list = []
    for filename in images:
        image_list.append(imageio.imread(filename))
    
    # Save the images as a GIF
    # Check if a GIF with the same name already exists and delete it if it does
    if os.path.exists(output_filename):
        os.remove(output_filename)
    # Save the new GIF
    imageio.mimsave(output_filename, image_list, duration=duration, loop=0)  # loop=0 makes it repeat indefinitely
    print(f"Repeating GIF created successfully: {output_filename}")

    gif_path = 'output_filename.gif'
    
    if os.path.exists(gif_path):
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', gif_path))
        elif platform.system() == 'Windows':
            os.startfile(gif_path)
        else:  # Linux and other Unix-like systems
            subprocess.call(('xdg-open', gif_path))
    else:
        print(f"Error: The GIF file was not found at {gif_path}")
