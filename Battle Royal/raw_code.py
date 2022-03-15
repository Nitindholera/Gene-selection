from math import ceil
from random import random
from re import A
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from mealpy.optimizer import Optimizer

class BaseBRO(Optimizer):
    """
        My best version of: Battle Royale Optimization (BRO)
            (Battle royale optimization algorithm)
        Link:
            https://doi.org/10.1007/s00521-020-05004-4
    """
    ID_DAM = 2

    def __init__(self, problem, epoch=10000, pop_size=100, threshold=3, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.threshold = threshold

        ## Dynamic variable
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = round(self.epoch / shrink)

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        damage = 0
        return [position, fitness, damage]

    def find_argmin_distance(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = np.reshape(dist_list, (-1))
        idx = np.argmin(dist_list[np.nonzero(dist_list)])
        return idx

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_argmin_distance(self.pop[i][self.ID_POS], self.pop)
            if self.compare_agent(self.pop[i], self.pop[j]):
                ## Update Winner based on global best solution
                pos_new = self.pop[i][self.ID_POS] + np.random.uniform() * \
                          np.mean(np.array([self.pop[i][self.ID_POS], self.g_best[self.ID_POS]]), axis=0)
                fit_new = self.get_fitness_position(pos_new)
                dam_new = self.pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                self.pop[i] = [pos_new, fit_new, dam_new]
                ## Update Loser
                if self.pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = np.random.uniform() * (np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS]) -
                                                       np.minimum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])) + \
                                          np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])
                    dam_new = self.pop[j][self.ID_DAM] + 1

                    self.pop[j][self.ID_FIT] = self.get_fitness_position(self.pop[j][self.ID_POS])
                else:  ## Loser dead and respawn again
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
                    dam_new = 0
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                self.pop[j] = [pos_new, fit_new, dam_new]
                nfe_epoch += 2
            else:
                ## Update Loser by following position of Winner
                self.pop[i] = deepcopy(self.pop[j])
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.pop[j][self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                dam_new = 0
                self.pop[j] = [pos_new, fit_new, dam_new]
                nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([self.pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best[self.ID_POS] - pos_std
            ub = self.g_best[self.ID_POS] + pos_std
            self.problem.lb = np.clip(lb, self.problem.lb, self.problem.ub)
            self.problem.ub = np.clip(ub, self.problem.lb, self.problem.ub)
            self.dyn_delta += np.round(self.dyn_delta / 2)

    def after_evolve(self, epoch):
        mutation_fraction = 0.05
        num_iterations = int(mutation_fraction*self.pop_size)
        sigma_initial = 100
        fac = ceil((epoch+1)/10)
        sigma = sigma_initial/pow((self.problem.ub[0] - self.problem.lb[0]),int(epoch/fac))
        for i in range(num_iterations):
            random_player = int(random()*self.pop_size)
            self.mutate(random_player, sigma)
            
    def mutate(self, player, sigma):
        #gaussian mutation
        #print(self.pop[player][self.ID_POS][2])
        chromosome = np.random.randint(0,len(self.pop[player][self.ID_POS]))
        mutated_pos = self.pop[player][self.ID_POS]
        mutated_pos[chromosome] = mutated_pos[chromosome] + sigma*np.random.normal(0,1)
        mutated_fitness = self.get_fitness_position(mutated_pos)
        mutated_dam = self.pop[player][self.ID_DAM]
        mutated_player = [mutated_pos, mutated_fitness, mutated_dam]
        
        if self.compare_agent(mutated_player, self.pop[player]):
            self.pop[player] = mutated_player

        
        

def objective_fun(x):
    return (x[0]-1)**2+(x[1]-2)**2+(x[2]-3)**2

problem_dict = {
    "obj_func": objective_fun,
    "lb": [-10,-10,-10],
    "ub": [10,10,10],
    "minmax": "min",
    "verbose": False
}

optimizer = BaseBRO(problem=problem_dict, epoch=100, pop_size=1000, threshold= 3 )

best_solution, best_fitness = optimizer.solve()
print("best solution is: ",best_solution)
print("best fittness is: ",best_fitness)