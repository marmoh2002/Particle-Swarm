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

## List of Benchmark Functions Supported
1. ackley
2. beale
3. booth
4. bukin
5. cross_in_tray
6. easom
7. eggholder
8. goldstein_price
9. griewank
10. holder_table
11. levi
12. levy
13. matyas
14. mccormick
15. michalewicz
16. rastrigin
17. rosenbrock
18. schaffer_n2
19. schaffer_n4
20. schwefel
21. sphere
22. styblinski_tang
23. three_hump_camel
24. trid
25. Input custom function


## Requirements

- NumPy
- Matplotlib
