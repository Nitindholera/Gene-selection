import numpy as np
from mealpy.optimizer import Optimizer
from copy import deepcopy
from scipy.spatial.distance import cdist

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
        sorted_pop, g1 = self.get_global_best_solution(self.pop) #returns sorted population and the best solution
        g2 = sorted_pop[1] #2nd best player
        
        g1_children = [] # will contain children from g1
        g2_children = [] # will contain children from g2
        eta = 3          # distribution index used during SBC

        for idx, parent in enumerate(sorted_pop):
            if idx != 0:
                u = np.random.random() # generates a random number between [0,1)
                
                if u<=0.5:
                    beta = pow(2*u, 1/(eta+1))
                else:
                    beta = pow(1/(2*(1-u)), 1/(eta+1))

                child1, child2 = self.generate_children_SBC(parent, g1, beta)
                g1_children.append(child1)
                g1_children.append(child2) #generating and inserting the children in the list for g1
            
            if idx!=1:
                u = np.random.random() # generates a random number between [0,1)
                
                if u<=0.5:
                    beta = pow(2*u, 1/(eta+1))
                else:
                    beta = pow(1/(2*(1-u)), 1/(eta+1))

                child1, child2 = self.generate_children_SBC(parent, g2, beta)
                g2_children.append(child1)
                g2_children.append(child2) #generating and inserting the children in the list for g2

        after_pop = self.pop + g1_children + g2_children #concanated population containing initial population and the children
        if self.problem.minmax == "min": #sort the after_pop to chose best n players in population for next iteration
            after_sorted_pop = sorted(after_pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR])
        else:
            after_sorted_pop = sorted(after_pop, key=lambda agent: agent[self.ID_FIT][self.ID_TAR], reverse=True)

        self.pop = after_sorted_pop[:self.pop_size] #chose best n players in population for next

    def generate_children_SBC(self, parent1, parent2, beta):
        temp1 = [i*(1+beta) for i in parent1[self.ID_POS]]
        temp2 = [j*(1-beta) for j in parent2[self.ID_POS]]
        child1_pos = np.array([(temp1[i]+temp2[i])*0.5 for i in range(len(temp1))]) #generate child1 position according to SBC formula
        child1_fit = self.get_fitness_position(child1_pos) #finding fitness from position
        child1_dam = 0

        temp3 = [i*(1-beta) for i in parent1[self.ID_POS]]
        temp4 = [j*(1+beta) for j in parent2[self.ID_POS]]
        child2_pos = np.array([0.5*(temp3[i]+temp4[i]) for i in range(len(temp3))]) #generate child2 position according to SBC formula
        child2_fit = self.get_fitness_position(child2_pos) ##finding fitness from position
        child2_dam = 0

        child1 = [child1_pos, child1_fit, child1_dam]
        child2 = [child2_pos, child2_fit, child2_dam]

        return child1, child2


def objective_fun(x):
    return (x[0]-1)**2+(x[1]-2)**2+(x[2]-3)**2

problem_dict = {
    "obj_func": objective_fun,
    "lb": [-10,-10,-10],
    "ub": [10,10,10],
    "minmax": "min",
    "verbose": True
}

optimizer = BaseBRO(problem=problem_dict, epoch=100, pop_size=100, threshold= 3 )

best_solution, best_fitness = optimizer.solve()
print("best solution is: ",best_solution)
print("best fittness is: ",best_fitness)