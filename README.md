# Particle Swarm Optimization (PSO) Implementation

This project implements and visualizes the Particle Swarm Optimization algorithm for various benchmark functions.

## Files

1. `psalg.py`: Core implementation of the PSO algorithm.

3. `funcs.py`: Contains 27 benchmark functions used for testing the PSO algorithm, as well as a custom function parser.

4. `check_implementation.py`: Evaluates the algorithm's performance by running it for a specific number of iterations. Calculates mean and standard deviation of results.

5. `plotted_pso.py`: Visualizes the benchmark function with swarm particles superimposed, showing the state before and after running the algorithm.

6. `drw.py`: Contains functions for visualizing the Particle Swarm Optimization (PSO) process and creating animated GIFs of the optimization.

7. `animate.py`: Creates animated visualizations of the PSO process, allowing users to choose benchmark functions or input custom functions for optimization.

## Usage

1. Run `plotted_pso.py` to visualize the PSO algorithm on a chosen benchmark function:
   ```
   python plotted_pso.py
   ```

2. Use `check_implementation.py` to evaluate the algorithm's performance:
   ```
   python check_implementation.py
   ```
## List of Benchmark Functions Supported
1. ackley
2. beale
3. booth
4. bukin
5. cross_in_tray
6. drop_wave
7. easom
8. eggholder
9. goldstein_price
10. griewank
11. himmelblau
12. holder_table
13. langermann
14. levy
15. matyas
16. mccormick
17. michalewicz
18. rastrigin
19. rosenbrock
20. schaffer_n2
21. schaffer_n4
22. schwefel
23. shubert
24. sphere
25. styblinski_tang
26. three_hump_camel
27. trid
28. Input custom function

## Requirements

- NumPy
- Matplotlib
