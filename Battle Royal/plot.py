from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('attribute.csv')

# accur = plt.plot(df['n_iteration'],df['max_accur'] )
# plt.ylabel('accuracy')
# plt.xlabel('Iteration')
# plt.savefig('Figures/accuracy.png')

# fit = plt.plot(df['n_iteration'],df['min_fitness'] )
# plt.ylabel('fitness')
# plt.xlabel('Iteration')
# plt.savefig('Figures/fitness.png')

rft = plt.plot(df['n_iteration'],df['rft'] )
plt.ylabel('relative number of features')
plt.xlabel('Iteration')
plt.savefig('Figures/features.png')