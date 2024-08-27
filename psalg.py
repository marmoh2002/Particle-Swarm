import numpy as np
import time
import funcs as funcs

class Particle:
    def __init__(self, lb: np.ndarray, ub: np.ndarray, num_dimensions: int):
        self.position = np.random.uniform(lb, ub, num_dimensions)
        self.velocity = np.random.uniform(-1, 1, num_dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best: np.ndarray, w: float, c1: float, c2: float, vmax: np.ndarray, a: float):
        r1, r2 = np.random.rand(2)
        if np.random.rand() < a:
            self.velocity = self.velocity
        else:
            self.velocity = (w * self.velocity +
                             c1 * r1 * (self.best_position - self.position) +
                             c2 * r2 * (global_best - self.position))
        self.velocity = np.clip(self.velocity, -vmax, vmax)

    def update_position(self, global_best: np.ndarray, w: float, lb: np.ndarray, ub: np.ndarray):
        r3, r4 = np.random.rand(2)
        if np.random.rand() < 0.49:
            self.position = global_best + w * r3 * np.abs(global_best)
        else:
            self.position = global_best - w * r4 * np.abs(global_best)
        self.position = np.clip(self.position, lb, ub)

    def evaluate(self, objective_func):
        fitness = objective_func(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        return fitness

class ParticleSwarm:
    def __init__(self, objective_func, lb, ub, num_dimensions: int, options: dict = None):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_dimensions = num_dimensions
        
        if len(self.lb) == 1:
            self.lb = np.repeat(self.lb, self.num_dimensions)
        if len(self.ub) == 1:
            self.ub = np.repeat(self.ub, self.num_dimensions)
        
        assert len(self.lb) == self.num_dimensions, "Lower bounds must match the number of dimensions"
        assert len(self.ub) == self.num_dimensions, "Upper bounds must match the number of dimensions"
        
        # Initialize options with provided dictionary or an empty dict if None
        # This allows for default values to be set for unspecified options
        self.options = options if options is not None else {}
        # Set the swarm size from the options dictionary, defaulting to 50 if not specified
        self.swarm_size = self.options.get('SwarmSize', 50)
        print("swarm size: ", self.swarm_size)
        self.max_iterations = self.options.get('MaxIterations', 1000)
        self.inertia_constant = self.options.get('InertiaConstant', 0.5)
        self.velocity_pausing_coefficient = self.options.get('VelocityPausingCoefficient', 0.5)        
        self.particles = [Particle(self.lb, self.ub, self.num_dimensions) for _ in range(self.swarm_size)]
        self.global_best_position = np.random.uniform(self.lb, self.ub, self.num_dimensions)
        self.global_best_fitness = float('inf')
        # This sets the maximum velocity for each dimension as 20% of the range between upper and lower bounds
        self.vmax = 0.2 * (self.ub - self.lb)

    def optimize(self):
        print("Starting PSO optimization...")
        start_time = time.time()
        for iteration in range(self.max_iterations):
            for i, particle in enumerate(self.particles):
                fitness = particle.evaluate(self.objective_func)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    # print(f"Iteration {iteration + 1}, Particle {i + 1}: New global best fitness = {self.global_best_fitness}")

            if self._check_convergence():
                print(f"Converged after {iteration + 1} iterations.")
                break

            self._update_particles(iteration)
            
            # if (iteration + 1) % 10 == 0:
            #     print(f"Completed {iteration + 1} iterations. Current best fitness: {self.global_best_fitness}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Optimization completed. Best fitness: {self.global_best_fitness}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        return self.global_best_position, self.global_best_fitness, elapsed_time

    def _update_particles(self, iteration):
        w = self._calculate_inertia_weight(iteration)
        c1 = self.options.get('SelfAdjustmentWeight', 1.4)
        c2 = self.options.get('SocialAdjustmentWeight', 1.4)

        for particle in self.particles:
            particle.update_velocity(self.global_best_position, w, c1, c2, self.vmax, self.velocity_pausing_coefficient)
            particle.update_position(self.global_best_position, w, self.lb, self.ub)

    def _calculate_inertia_weight(self, iteration):
        b = self.inertia_constant
        t = iteration
        T = self.max_iterations
        return np.exp(-(b * t / T)) * b

    def _check_convergence(self):
        avg_fitness = np.mean([p.best_fitness for p in self.particles])
        tolerance = self.options.get('Tolerance', 1e-6)
        converged = abs(avg_fitness - self.global_best_fitness) < tolerance
        if converged:
            print(f"Convergence achieved. Average fitness: {avg_fitness}, Global best fitness: {self.global_best_fitness}")
        return converged

# # Example usage
# if __name__ == "__main__":
#     num_dimensions = int(input("Enter the number of dimensions: "))
#     use_default = input("Use default bounds (-5.12 to 5.12)? (y/n): ").lower() == 'y'
    
#     if use_default:
#         lb = [-5.12]*num_dimensions
#         ub = [5.12]*num_dimensions
#     else:
#         lb_input = input("Enter the lower bound(s) (comma-separated if multiple): ")
#         ub_input = input("Enter the upper bound(s) (comma-separated if multiple): ")
#         lb = [float(x) for x in lb_input.split(',')]
#         ub = [float(x) for x in ub_input.split(',')]
    
#     function_choice = input("Choose an objective function (1: Sphere, 2: Rastrigin): ")
    
#     if function_choice == '1':
#         objective_func = funcs.sphere
#     elif function_choice == '2':
#         objective_func = funcs.rastrigin
#     else:
#         raise ValueError("Invalid function choice")

#     pso = ParticleSwarm(objective_func, lb, ub, num_dimensions)
#     best_position, best_fitness, elapsed_time = pso.optimize()

#     print(f"Best position: {best_position}")
#     print(f"Best fitness: {best_fitness}")
#     print(f"Time elapsed: {elapsed_time:.2f} seconds")