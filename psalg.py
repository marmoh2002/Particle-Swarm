import numpy as np
import time
import funcs as funcs

class Particle:
    def __init__(self, lb: np.ndarray, ub: np.ndarray, num_dimensions: int, minimize: bool):
        self.position = np.random.uniform(lb, ub, num_dimensions)
        self.velocity = np.random.uniform(-1, 1, num_dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf') if minimize else float('-inf')

    def update_velocity(self, global_best: np.ndarray, w: float, c1: float, c2: float, vmax: np.ndarray):
        r1, r2 = np.random.rand(2)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best - self.position))
        self.velocity = np.clip(self.velocity, -vmax, vmax)

    def update_position(self, lb: np.ndarray, ub: np.ndarray):
        self.position += self.velocity
        self.position = np.clip(self.position, lb, ub)

    def evaluate(self, objective_func, minimize=True):
        fitness = objective_func(self.position)
        if minimize:
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.position.copy()
        else:
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.position.copy()
        return fitness

class ParticleSwarm:
    def __init__(self, objective_func, lb, ub, num_dimensions: int, options: dict = None, minimize=True):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_dimensions = num_dimensions
        self.minimize = minimize
        
        if len(self.lb) == 1:
            self.lb = np.repeat(self.lb, self.num_dimensions)
        if len(self.ub) == 1:
            self.ub = np.repeat(self.ub, self.num_dimensions)
        
        assert len(self.lb) == self.num_dimensions, "Lower bounds must match the number of dimensions"
        assert len(self.ub) == self.num_dimensions, "Upper bounds must match the number of dimensions"
        
        self.options = options if options is not None else {}
        self.swarm_size = self.options.get('SwarmSize', 50)
        print("swarm size: ", self.swarm_size)
        self.max_iterations = self.options.get('MaxIterations', 1000)
        self.w_start = self.options.get('InertiaStartWeight', 0.9)
        self.w_end = self.options.get('InertiaEndWeight', 0.4)
        self.c1 = self.options.get('SelfAdjustmentWeight', 2.0)
        self.c2 = self.options.get('SocialAdjustmentWeight', 2.0)
        self.particles = [Particle(self.lb, self.ub, self.num_dimensions, self.minimize) for _ in range(self.swarm_size)]
        self.global_best_position = np.random.uniform(self.lb, self.ub, self.num_dimensions)
        self.global_best_fitness = float('inf') if self.minimize else float('-inf')
        self.vmax = 0.2 * (self.ub - self.lb)

    def optimize(self):
        print(f"Starting PSO optimization... ({'minimization' if self.minimize else 'maximization'})")
        start_time = time.time()
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                fitness = particle.evaluate(self.objective_func, self.minimize)
                if self.minimize:
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle.position.copy()
                else:
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle.position.copy()

            if self._check_convergence():
                print(f"Converged after {iteration + 1} iterations.")
                break
            self._update_particles(iteration)
            

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Optimization completed. Best fitness: {self.global_best_fitness}")
        print(f"Position of best fitness: {self.global_best_position}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        return self.global_best_position, self.global_best_fitness, elapsed_time

    def _update_particles(self, iteration):
        w = self._calculate_inertia_weight(iteration)
        for particle in self.particles:
            particle.update_velocity(self.global_best_position, w, self.c1, self.c2, self.vmax)
            particle.update_position(self.lb, self.ub)

    def _calculate_inertia_weight(self, iteration):
        return self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iterations)

    def _check_convergence(self):
        positions = np.array([p.position for p in self.particles])
        position_range = np.max(positions, axis=0) - np.min(positions, axis=0)
        fitness_range = np.max([p.best_fitness for p in self.particles]) - np.min([p.best_fitness for p in self.particles])
        tolerance = self.options.get('Tolerance', 1e-6)
        converged = np.all(position_range < tolerance) and fitness_range < tolerance
        return converged
