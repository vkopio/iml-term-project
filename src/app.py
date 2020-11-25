import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    

def app():
    #print(np.pi)
    npf = preprosessing(npf_train)
    X = npf.drop(['class2','Ia','Ib','II','nonevent'],1)
    y_class2 = npf['class2']
    y_class4 = npf[['Ia','Ib','II','nonevent']]

    X_train, X_test, y_class2_train, y_class2_test = train_test_split(X, y_class2, train_size=0.8, random_state=1)


    return 'Hello, World!'
