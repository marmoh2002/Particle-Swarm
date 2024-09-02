# Particle Swarm Optimization (PSO) Implementation

This project implements and visualizes the Particle Swarm Optimization algorithm for various benchmark functions.

## Files

1. `psalg.py`: Core implementation of the PSO algorithm.

3. `check_implementation.py`: Evaluates the algorithm's performance by running it for a specific number of iterations. Calculates mean and standard deviation of results.

4. `funcs.py`: Contains 30 benchmark functions used for testing the PSO algorithm.

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
5. bukin_n6
6. cross_in_tray
7. drop_wave
8. easom
9. eggholder
10. functions_menu
11. goldstein_price
12. griewank
13. himmelblau
14. holder_table
15. langermann
16. levi
17. levy
18. matyas
19. mccormick
20. michalewicz
21. rastrigin
22. rosenbrock
23. schaffer_n2
24. schaffer_n4
25. schwefel
26. shubert
27. sphere
28. styblinski_tang
29. three_hump_camel
30. trid
31. Input custom function


## Requirements

- NumPy
- Matplotlib
