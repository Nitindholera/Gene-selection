import pandas as pd
import math
import random

def sigm(x):
    return 1/(1+math.exp(-1*x))

df = pd.read_csv("data_normal.csv", header= None)
rows, columns = df.shape[0], df.shape[1]
for i in range(columns-3):
    for j in range(rows):
        df[i][j] = sigm(df[i][j])
df.to_csv("i1.csv", index=None, header= None)
for i in range(1,101):
    file_name = 'i' + str(i) + 's' + str(i) + '.csv'
    dfc = df.copy(deep = False)
    x = random.uniform(sigm(0), sigm(1))
    d = dict()
    for i in range(columns - 3):
        s = 0
        for j in range(rows):
            if dfc[i][j] < x: dfc[i][j] = 0
            else: dfc[i][j] = 1
            s += dfc[i][j]
        d[i] = s
    d[columns-3], d[columns-2], d[columns-1] = "", "", ""
    dfc = dfc.append(d, ignore_index= True)
    dfc.to_csv(file_name, header= None, index= None)
