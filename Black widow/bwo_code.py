from bwo import minimize
from landscapes.single_objective import sphere

fbest,xbest = minimize(sphere, dof = 5)
print(fbest)
print(xbest)

