##split dataset

import pandas as pd
import pandas as pd
import os
import re
import numpy as np
from scipy import stats
from pandas import set_option
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"/Users/sanjirsalsabil/Desktop/bank.csv")
df.head()

from sklearn.model_selection import train_test_split

x = variable[['age', 'marital']]
y = variable['education']

import sklearn.model_selection as model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.65,test_size=0.35, 
                                                                    random_state=101)
print ("x_train: ", x_train)
print ("y_train: ", y_train)
print("x_test: ", x_test)
print ("y_test: ", y_test)