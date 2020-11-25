import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
from sklearn.linear_model import *
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

npf_train = pd.read_csv("data/npf_train.csv")
npf_test = pd.read_csv("data/npf_test_hidden.csv")


def preprosessing(npf):
    '''Preprosessing function for npf_*.csv files'''

    # Making the binary classifier class2 where event = 1, nonevent = 0
    npf['class2'] = [0 if x == 'nonevent' else 1 for x in npf['class4']]

    # Making dummy variables from class4 variables 'nonevent','Ia','Ib' and 'II' 
    dummies = pd.get_dummies(npf['class4'])
    npf = npf.drop('class4',axis=1)
    npf = npf.join(dummies)

    # Dropping features 'partlybad','id' and 'date' because we won't need them. Feature 'partlybad' was only False 
    npf = npf.drop(['date','id','partlybad'],axis=1)

    return npf

def linReg(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test), model

def logReg(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]

    #return roc_auc_score(y_test, y_hat) 
    return model.score(X_test, y_test), model

def gaussianNB(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test)

def best_feature_columns(X_train, y_train, n):
    '''Return n best feature columns'''
    select = sklearn.feature_selection.SelectKBest(k=20)
    selected_features = select.fit(X_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X_train.columns[i] for i in indices_selected]

    return colnames_selected


def app():
    #print(np.pi)
    npf = preprosessing(npf_train)
    X = npf.drop(['class2','Ia','Ib','II','nonevent'],1)
    y_class2 = npf['class2']
    y_class4 = npf[['Ia','Ib','II','nonevent']]

    # Test and training sets for binary classification
    X_train, X_test, y_class2_train, y_class2_test = train_test_split(X, y_class2, train_size=0.8, random_state=1)
    

    ### Testing ###

    #print(linReg(X_train, y_class2_train, X_test, y_class2_test))
    #print(logReg(X_train, y_class2_train, X_test, y_class2_test))
    print(gaussianNB(X_train, y_class2_train, X_test, y_class2_test))

    # Dropping stds

    X_means = X.drop([c for c in X.columns if 'std' in c],axis=1)
    X_train, X_test, y_class2_train, y_class2_test = train_test_split(X_means, y_class2, train_size=0.8, random_state=1)
    #print(linReg(X_train, y_class2_train, X_test, y_class2_test))
    #print(logReg(X_train, y_class2_train, X_test, y_class2_test))
    print(gaussianNB(X_train, y_class2_train, X_test, y_class2_test))







    
    # Selecting 20 best features
    colnames_selected = best_feature_columns(X_train, y_class2_train, 20)
    X_train_selected = X_train[colnames_selected]
    X_test_selected = X_test[colnames_selected]


    



    return 'Hello, World!'

# Models which could be tested
#   - Linear regression
#   - Lasso regression
#   - SVM
#   - Naive Bayesian
#   - k-NN
