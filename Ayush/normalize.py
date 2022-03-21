import pandas as pd
import sys
n = len(sys.argv)
if n!=2:
    sys.exit("Invalid arguments")
df = pd.read_table(sys.argv[1], delimiter=',', header=None)
df.to_csv('data.csv', header=None, index=None)
rows, columns = df.shape[0], df.shape[1]
for i in range(columns-1):
    df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())
df.to_csv("data_normal.csv", header=None, index=None)