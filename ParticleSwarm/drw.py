import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotted_pso import plot_3d_function
import os
import imageio
import glob

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
    ax.scatter(positions[:, 0], positions[:, 1], z, color='magenta', s=35, label='positions')
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
    plt.savefig(f'pso_figures/iteration_{run_num}.png')
    # plt.pause(0.001)  # 100 milliseconds
    plt.close(fig)
       


def create_gif_from_images(folder_path, output_filename='animation.gif', duration=1.5):
    """
    Create a GIF from a folder of images.
    
    :param folder_path: Path to the folder containing the images
    :param output_filename: Name of the output GIF file
    :param duration: Duration of each frame in the GIF (in seconds)
    """
    # Get list of PNG files in the folder, sorted by name
    images = sorted(glob.glob(f"{folder_path}/*.png"))
    
    # Read in all the images
    image_list = []
    for filename in images:
        image_list.append(imageio.imread(filename))
    
    # Save the images as a GIF
    # Check if a GIF with the same name already exists and delete it if it does
    if os.path.exists(output_filename):
        os.remove(output_filename)
    # Save the new GIF
    imageio.mimsave(output_filename, image_list, duration=duration)
    print(f"GIF created successfully: {output_filename}")

# Example usage:
# create_gif_from_images(path, 'pso_animation.gif', 0.5)
