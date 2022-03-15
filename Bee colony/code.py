from mealpy.swarm_based import ABC
import numpy as np

def objective_fun(x):
    return (x[0]-1)**2+(x[1]-2)**2

problem_dict = {
    "obj_func": objective_fun,
    "lb": [-10]*2,
    "ub": [10]*2,
    "minmax": "min",
    "verbose": True
}

optimizer = ABC.BaseABC(problem=problem_dict, epoch= 1000, pop_size=100,)
best_solution, best_fitness = optimizer.solve()
print("best solution:", best_solution)
print("best fittness: ",best_fitness)
#print(optimizer.solution)