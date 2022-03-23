import pandas as pd
df = pd.read_csv("data_normal.csv", header= None)
rows, cols = df.shape[0], df.shape[1]
l = []
for i in range(cols-3):
    l.append([i,0])
for i in range(1,101):
    file_name = "i" + str(i) + "s" + str(i) + ".csv"
    dfi = pd.read_csv(file_name, header= None)
    for j in range(cols-3):
        l[j][1] += dfi[j][rows-1]
l.sort(reverse= True)
df_final = pd.DataFrame()
cols1 = (cols-3)//2
for i in range(cols1):
    df_final[i] = df[l[i][0]]
df_final[cols1] = df[cols-3]
df_final[cols1+1] = df[cols-2]
df_final[cols1+2] = df[cols-1]
df_final.to_csv("50_attributes.csv", index= None, header= None)
