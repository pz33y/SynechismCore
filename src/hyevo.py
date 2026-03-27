"""
HyEvo — Multi-Island Genetic Algorithm for Hyperparameter Search
================================================================
Optimizes: phi_base, alpha, R, and lr_scale.
Implements 4 islands, 8 individuals each, 20 generations, ring migration.

Author: Paul E. Harris IV — SynechismCore v20.1
"""

import torch
import numpy as np

class MultiIslandGA:
    """Multi-island genetic algorithm for optimizing Synechism parameters."""
    def __init__(self, n_islands=4, pop_size=8, n_generations=20, migration_interval=5):
        self.n_islands = n_islands
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.migration_interval = migration_interval
        
        # Hyperparameter search space
        self.bounds = {
            'phi_base': (1.0, 2.0),
            'alpha': (0.01, 1.0),
            'R': (0.5, 2.0),
            'lr_scale': (0.1, 10.0)
        }
        
        # Initialize islands
        self.islands = [self._init_population() for _ in range(n_islands)]

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {k: np.random.uniform(v[0], v[1]) for k, v in self.bounds.items()}
            pop.append({'params': ind, 'fitness': float('-inf')})
        return pop

    def mutate(self, params, rate=0.1):
        """Gaussian mutation."""
        new_params = params.copy()
        for k, v in self.bounds.items():
            if np.random.rand() < rate:
                new_params[k] = np.clip(
                    new_params[k] + np.random.normal(0, (v[1]-v[0])*0.1),
                    v[0], v[1]
                )
        return new_params

    def crossover(self, p1, p2):
        """Uniform crossover."""
        child = {}
        for k in self.bounds.keys():
            child[k] = p1[k] if np.random.rand() < 0.5 else p2[k]
        return child

    def step(self, generation, fitness_func):
        """Single generation step across all islands."""
        for i, island in enumerate(self.islands):
            # Evaluate fitness
            for ind in island:
                if ind['fitness'] == float('-inf'):
                    ind['fitness'] = fitness_func(ind['params'])
            
            # Tournament selection and reproduction
            new_pop = []
            for _ in range(self.pop_size):
                # Select parents
                idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = island[idx1] if island[idx1]['fitness'] > island[idx2]['fitness'] else island[idx2]
                
                idx3, idx4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = island[idx3] if island[idx3]['fitness'] > island[idx4]['fitness'] else island[idx4]
                
                # Crossover and mutate
                child_params = self.crossover(p1['params'], p2['params'])
                child_params = self.mutate(child_params)
                new_pop.append({'params': child_params, 'fitness': float('-inf')})
            
            self.islands[i] = new_pop
            
        # Migration (ring topology)
        if (generation + 1) % self.migration_interval == 0:
            for i in range(self.n_islands):
                source = i
                target = (i + 1) % self.n_islands
                # Migrate the best individual
                best_idx = np.argmax([ind['fitness'] for ind in self.islands[source]])
                self.islands[target][0] = self.islands[source][best_idx].copy()
                
    def get_best(self):
        all_inds = [ind for island in self.islands for ind in island]
        best_idx = np.argmax([ind['fitness'] for ind in all_inds])
        return all_inds[best_idx]
