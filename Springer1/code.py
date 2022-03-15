import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC          
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('Springer1_Dataset/pone.0246039.s001.csv')
X, y = data.iloc[:,1:-1], data.iloc[:,[-1]]
y['Response'] = y['Response'].map({'normal' : 0, 'tumar' : 1})
X = X.to_numpy()
y = y.to_numpy()

g = X.transpose()
kmean = KMeans(n_clusters=80)
kmean.fit(g)    
representative_genes = np.zeros(shape=(80,g.shape[1]))

dist = np.array([-1 for x in range(80)])
for i in range(1,81):
    for j in range(g.shape[0]):
        if kmean.labels_[j] == i:
            #print(abs(np.dot(kmean.cluster_centers_[i-1], g.iloc[j])))
            if dist[i-1] == -1:
                dist[i-1] =abs(np.dot(kmean.cluster_centers_[i-1], g[j]))
                representative_genes[i-1] = g[j]
            elif abs(np.dot(kmean.cluster_centers_[i-1], g[j]))<dist[i-1]:
                dist[i-1]=abs(np.dot(kmean.cluster_centers_[i-1], g[j]))
                representative_genes[i-1] = g[j]

X = representative_genes.T

svm = LinearSVC(C=10)
svm_rfe = RFE(estimator=svm,n_features_to_select=7)
svm_rfe = svm_rfe.fit(X,y)

X = svm_rfe.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4722)
model = LinearSVC(C=10)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))

