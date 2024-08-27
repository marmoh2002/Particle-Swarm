files:

1- psalg.py: implementation of the pso algorithm (still needs changes, implementing group pso instead of individual pso)

2- check_implementation.py: runs the algorithm for a specific number of iterations to calculate the mean and std of the obtained results (needs some fine tuning since currently all it does is automatically evaluate the sphere and rastrigin functions upon running)

3- funcs.py: contains the benchmark functions needed

4- plotted_pso.py: plots the benchmark function with the swarm particles superimposed on it, before and after running the algorithm