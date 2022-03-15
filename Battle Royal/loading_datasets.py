import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, svm
import sklearn
from sklearn.model_selection import KFold, train_test_split
import minepy

german_df = pd.read_csv("Datasets/Diabetic/messidor_features.csv", sep=",", )

min_max_scaler = preprocessing.MinMaxScaler()
d = min_max_scaler.fit_transform(german_df.iloc[:,:-1].transpose())
names = german_df.columns[:-1]

scaled_german_df = pd.DataFrame(d.transpose(), columns=names) #row wise min-maxscaled
X = scaled_german_df.iloc[:,:]
y = german_df.iloc[:,-1]

accuracy = np.zeros(5)

clf = svm.SVC(kernel='linear')

kf = KFold(n_splits=5, shuffle=True)

idx = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #print("Accuracy",metrics.accuracy_score(y_test, y_pred))
    accuracy[idx]=(metrics.accuracy_score(y_test, y_pred))
    idx+=1
    
    
print("min accuracy", accuracy.min())
print("max accuracy", accuracy.max())
print("avg accuracy", accuracy.mean())

#print(clf.coef_) #prints weights of linear svm

mine = minepy.MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(X.iloc[:,1], y)
mic = mine.mic()
print(mic)