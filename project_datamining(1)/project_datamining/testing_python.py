import matplotlib.pyplot as plt
import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
from sklearn import tree
import sklearn.model_selection


data=pd.read_excel('Raisin_Dataset/Raisin_Dataset.xlsx')
data.to_numpy()
data=data.values
X = data[:,:-1]


atDict = {}
Attributes = ['Area','MajorAxisLength', 'MinorAxisLength','Eccentricity', 'ConvexArea', 'Extent', 'Perimeter' ]
for i in range(len(Attributes)):
    atDict[Attributes[i]] = data[:,i]
#print(atDict['MajorAxisLength'])
atDict['class'] = data[:,7]

bools = (atDict['MajorAxisLength'] > 422.423) & (atDict['MajorAxisLength'] <= 452.894) & (atDict['ConvexArea'] <= 77974) 

    
    
y_bools = atDict['class'] != "Kecimen"
y = []
for i in range(len(y_bools)):
    if(y_bools[i]):
        y.append(1)
    else:
        y.append(0)
y = np.array(y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

skf = sklearn.model_selection.StratifiedKFold(n_splits=10)

clf = RandomForestClassifier( max_depth= 8, random_state= 23)
accuracies_list =[]
fpr_list, tpr_list = [], []

for train, test in skf.split(X,y):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
      
    clf = clf.fit(X_train, y_train)
    preditcted_probabilities = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, preditcted_probabilities[:,1])
    plt.plot(fpr, tpr)
plt.savefig("myplot.png")


