# Particle Swarm Optimization (PSO) Implementation

This project implements and visualizes the Particle Swarm Optimization algorithm for various benchmark functions.

## Files

1. `psalg.py`: Core implementation of the PSO algorithm.

3. `check_implementation.py`: Evaluates the algorithm's performance by running it for a specific number of iterations. Calculates mean and standard deviation of results.

4. `funcs.py`: Contains 12 benchmark functions used for testing the PSO algorithm.

5. `plotted_pso.py`: Visualizes the benchmark function with swarm particles superimposed, showing the state before and after running the algorithm.

## Usage

1. Run `plotted_pso.py` to visualize the PSO algorithm on a chosen benchmark function:
   ```
   python plotted_pso.py
   ```

2. Use `check_implementation.py` to evaluate the algorithm's performance:
   ```
   python check_implementation.py
   ```

## TODO

- revise `psalg.py` functionality.

## Requirements

- NumPy
- Matplotlib
