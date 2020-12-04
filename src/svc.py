import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC

def preprocess(npf):
    # Binary variable for event vs nonevent
    npf['class2'] = [0 if x == 'nonevent' else 1 for x in npf['class4']]
    
    # Making dummy variables from class4 variables 'nonevent','Ia','Ib' and 'II'
    dummies = pd.get_dummies(npf['class4'])
    npf = npf.drop('class4',axis=1)
    npf = npf.join(dummies)

    # Dropping features 'partlybad','id' and 'date' because we won't need them. Feature 'partlybad' was only False
    npf = npf.drop(['date','id','partlybad'],axis=1)
    X = npf.drop(['class2','Ia','Ib','II','nonevent'],1)
    X_means = X.drop([c for c in X.columns if 'std' in c],axis=1)
    
    # Remove .means from all column names
    cols = [col[:-5] for col in X_means.columns]
    
    # Normalize
    X_means_np = scale(X_means)
    
    df = pd.DataFrame(X_means_np, columns=cols)
    
    df["class2"] = npf["class2"]
    
    return df

def preprocess_test(npf):
    X = npf.drop(['date','id','partlybad', 'class4'],axis=1)
    X_means = X.drop([c for c in X.columns if 'std' in c],axis=1)
    
    cols = [col[:-5] for col in X_means.columns]
    
    X_means_np = scale(X_means)
    
    df = pd.DataFrame(X_means_np, columns=cols)
    
    return df

npf_train = pd.read_csv("data/npf_train.csv")
npf_test = pd.read_csv("data/npf_test_hidden.csv")

c = 2.80135676119887

npf = preprocess(npf_train)
x = npf.iloc[:,:-2]
y = npf.iloc[:,-1]

npft = preprocess_test(npf_test)
x_test = npft.iloc[:,:-1]
y_test = npft.iloc[:,-1]

clf = SVC(C=c, probability=True)
clf.fit(x, y)

print(clf.predict_proba(x_test))
