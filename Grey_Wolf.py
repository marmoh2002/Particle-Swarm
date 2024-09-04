import numpy as np
import matplotlib.pyplot as plt
import funcs
import time

# Grey Wolf Optimizer (GWO)
def GWO(objective_function, dim, n_wolves, max_iter, lb, ub, visualize=False,patience_choice=True, stop_threshold=1e-6, patience=10):
    # Initialize alpha, beta, and delta positions
    Alpha_pos = np.zeros(dim)
    Beta_pos = np.zeros(dim)
    Delta_pos = np.zeros(dim)
    
    # Initialize alpha, beta, and delta fitness values
    Alpha_score = float('inf')
    Beta_score = float('inf')
    Delta_score = float('inf')
    
    # Initialize the positions of the wolves
    Wolves = np.random.uniform(lb, ub, (n_wolves, dim))
    
    if visualize:
        # Set up the plot for the function surface
        plt.ion()
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid for the objective function
        x = np.linspace(lb, ub, 400)
        y = np.linspace(lb, ub, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
       
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
               Z[i, j] = objective_function(np.array([X[i, j], Y[i, j]]))
        
        # Plot the function surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # Plot initial positions of wolves
        scat = ax.scatter(Wolves[:, 0], Wolves[:, 1], [objective_function(Wolves[i]) for i in range(n_wolves)], 
                          s=30, c='blue', marker='o', label='Wolves')
        alpha_scat = ax.scatter(Alpha_pos[0], Alpha_pos[1], objective_function(Alpha_pos), 
                                s=100, c='red', marker='x', label='Alpha')
        ax.set_title('Grey Wolf Optimizer')
        ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Objective Function Value')

    best_fitnesses = []
    all_fitness_values = []  # To store all fitness values

    # To monitor improvement and implement early stopping
    no_improvement_count = 0
    best_fitness_history = []

    # Main loop
    for iter in range(max_iter):
        iteration_fitness_values = []  # To store fitness values for this iteration

        for i in range(n_wolves):
            # Calculate the fitness of each wolf
            fitness = objective_function(Wolves[i])
            iteration_fitness_values.append(fitness)
            
            # Update alpha, beta, and delta
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Wolves[i].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Wolves[i].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Wolves[i].copy()

        # Store fitness values for this iteration
        all_fitness_values.extend(iteration_fitness_values)

        # Update the positions of the wolves
        a = 2 - iter * (2 / max_iter)  # Decrease a linearly from 2 to 0
        
        for i in range(n_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Wolves[i][j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Wolves[i][j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Wolves[i][j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                # Update the position of the current wolf
                Wolves[i][j] = (X1 + X2 + X3) / 3
        
        # Ensure wolves stay within bounds
        Wolves = np.clip(Wolves, lb, ub)
        
        if visualize:
            # Update the plot
            scat._offsets3d = (Wolves[:, 0], Wolves[:, 1], [objective_function(Wolves[i]) for i in range(n_wolves)])  # Update wolf positions
            alpha_scat._offsets3d = ([Alpha_pos[0]], [Alpha_pos[1]], [objective_function(Alpha_pos)])  # Update alpha position
            ax.set_title(f'Grey Wolf Optimizer (Iteration {iter+1})\nBest Fitness: {Alpha_score}')
            plt.draw()
            plt.pause(0.1)
        
        best_fitnesses.append(Alpha_score)
        
        if patience_choice:
            # Early stopping condition
            if iter > 0 and abs(best_fitnesses[-1] - best_fitnesses[-2]) < stop_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {iter+1} due to insufficient improvement.")
                break
    
    if visualize:
        plt.ioff()
        plt.show()
        
        # Plot the convergence curve
        plt.plot(best_fitnesses, 'r-', label='Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Convergence Curve')
        plt.legend()
        plt.show()
    
    return Alpha_pos, Alpha_score

def Mean_Variance_calc(best_scores=[]):  
    mean_score = np.mean(best_scores)  
    std_dev_score = np.std(best_scores) 
    print(f"\nMean of runs: {mean_score}")
    print(f"Standard deviation: {std_dev_score}")
    print(f"The optimal solution of each run: {best_scores}") 
     
    # Visualization of the best scores  
    plt.plot(best_scores, 'o-', label='Best Scores from Each Run')  
    plt.xlabel('Run Number')  
    plt.ylabel('Best Score')  
    plt.title('Best Scores from Multiple GWO Runs')  
    plt.legend()  
    plt.show()
    
    return 0

if __name__ == '__main__':
    # Parameters
    dim = 2           # Dimension of the problem (reduced to 2 for visualization)
    best_scores = []  # Array to store best scores each time we run the program
    available_functions = [func for func in dir(funcs) if callable(getattr(funcs, func)) and not func.startswith("__")]
    print("Available functions:")
    for i, func in enumerate(available_functions, 1):
        print(f"{i}. {func}")
    
    while True:
        choice = input("Choose a function to minimize (enter number or '0' to exit): ")
        
        if choice == '0':
            print("Exiting...")
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_functions):
                func_name = available_functions[index]
                objective_function = getattr(funcs, func_name)
                print(f"Minimizing {func_name} function...")
                
                visualization_choice = input("Do you want to visualize the function? (yes/no): ").strip().lower()
                visualize = visualization_choice == 'yes'

                patience_choice = input("Do you want to set a patience value for early stopping? (yes/no): ").strip().lower()
                patience_choice = patience_choice == 'yes'
                

                
                n_wolves = int(input("Enter number of wolves (default is 100): "))
                max_iter = int(input("Enter maximum number of iterations (default is 1000): "))
                n_runs = int(input("Enter number of runs (default is 10): "))
                lb = float(input("Enter lower bound of the search space (default is -1000): "))
                ub = float(input("Enter upper bound of the search space (default is 1000): "))
                
                if patience_choice:
                    patience = int(input("Enter the number of iterations with minimal improvement before stopping (default is 10): "))
                    stop_threshold = float(input("Enter the threshold for early stopping (default is 1e-6): "))
                else:
                    patience = max_iter
                    stop_threshold = 1e-6
                for _ in range(n_runs):
                    start_time = time.time()  # Start the timer
                    best_pos, best_score = GWO(objective_function, dim, n_wolves, max_iter, lb, ub, visualize=visualize, patience_choice=patience_choice, patience=patience,stop_threshold=stop_threshold)
                    elapsed_time = time.time() - start_time  # Calculate the elapsed time
                    
                    print(f"\nPosition of alpha wolf: {best_pos}")
                    print(f"Best score: {best_score}")
                    print(f"Time elapsed for this run: {elapsed_time:.4f} seconds")
                    best_scores.append(best_score)
                
            Mean_Variance_calc(best_scores)  # Determine the mean and the standard deviation of optimal solutions (alpha wolf) obtained each run
            best_scores = []  # Reset the best scores for the next run
        except Exception as e:
            print(f"Invalid choice or error occurred: {e}. Please try again.")
