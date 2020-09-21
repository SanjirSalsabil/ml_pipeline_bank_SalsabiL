##Test pipeline

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

pwd

df=pd.read_excel('bank_testing.csv')

df.info()
df.columns
df=df.dropna()


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(X)
print(y)


df[df['marital'].isnull()]
df[df['education'].isnull()]
df[df['age'].isnull()]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

dummies = pd.get_dummies(df.education)
print(dummies)

df['education'] = pd.to_numeric(df['education'],errors='coerce')

dummies = pd.get_dummies(df.marital)
print(dummies)

df['marital'] = pd.to_numeric(df['marital'],errors='coerce')

merged = pd.concat([df,dummies], axis = 'columns')
merged

final = merged.drop(['campaign', 'previous', 'pdays','education','marital'], axis='columns')
final

from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn import preprocessing

def convert(df):
    number = preprocessing.LabelEncoder()
    df['ceducation'] = number.fit_transform(df['education'])
    df=df.fillna(-999) # fill holes with default value
    return df

import pandas as pd

Data = {'education', 'marital', 'age'}

df = pd.DataFrame(Data)
print (df)
print (df.dtypes)

import pandas as pd

Data = {'education': ['primary','secondary','tertiary'],
          'marital': ['divorced','married','single'],}

df = pd.DataFrame(Data)
df['marital'] = pd.to_numeric(df['marital'], errors='coerce')
df['education'] = pd.to_numeric(df['education'], errors='coerce')


print (df)
print(df.dtypes)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

print(y_train)
print(y_test)
