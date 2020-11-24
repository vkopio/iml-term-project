import numpy as np
import pandas as pd

npf_train = pd.read_csv("data/npf_train.csv")
npf_test = pd.read_csv("data/npf_test_hidden.csv")

def app():
    print(np.pi)

    return 'Hello, World!'
