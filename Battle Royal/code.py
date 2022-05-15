from mealpy.human_based import BRO
import numpy as np

def objective_fun(x):
    return (x[0]-1)**2+(x[1]-2)**2+(x[2]-3)**2

problem_dict = {
    "obj_func": objective_fun,
    "lb": [-10,-10,-10],
    "ub": [10,10,10],
    "minmax": "min",
    "verbose": False
}

# optimizer = BRO.BaseBRO(problem=problem_dict, epoch=100, pop_size=100, threshold= 3 )

# best_solution, best_fitness = optimizer.solve()
# print("best solution is: ",best_solution)
# print("best fittness is: ",best_fitness)
print(np.random.randint(0,15))