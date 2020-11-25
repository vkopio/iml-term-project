import numpy as np
import pandas as pd

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

    # Dropping features 'partlybad','id' and 'date' because we won't need them
    npf = npf.drop(['date','id','partlybad'],axis=1)

    return npf

    

def app():
    #print(np.pi)
    npf = preprosessing(npf_train)
    
    return 'Hello, World!'
