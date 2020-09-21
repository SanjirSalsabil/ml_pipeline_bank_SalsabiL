##Train pipeline

#Import

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

pwd

variable = pd.read_csv(r"/Users/sanjirsalsabil/Desktop/DSS Learning/ml_pipeline_bank_SalsabiL/data/bank_training.csv")
variable.head()

X = variable.iloc[:, :-1].values
y = variable.iloc[:, -1].values

print(X)
print(y)

#describe

variable.shape
variable.describe()

Counter(variable["Target"])
variable.info()
variable.columns

constant_imp=SimpleImputer(strategy='constant', fill_value=0)
list_constant=["education"]
variable[list_constant]=constant_imp.fit_transform(df[list_constant])


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

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
len(x_train)

len(y_test)

##Null_value_detection

variable.notnull().head()

variable.isnull().sum()

variable[variable['marital'].isnull()]

variable[variable['education'].isnull()]

variable[variable['age'].isnull()]

variable.shape
variable.columns

variable.dropna()
variable.dropna(how = 'any').shape
variable.dropna(how = 'all').shape
variable.dropna(subset = ['education', 'balance', 'month'],how='all').shape

variable.dropna(how='any', thresh=1).shape

##Visualization_plot


import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(variable['age'], variable['balance'], variable['duration'])
plt.hist(variable.age)
plt.show()
plt.hist(variable.balance)
plt.show()
plt.hist(variable.duration)
plt.show()

#detect_Outliers

lower_bound = 0.1
upper_bound = 0.95
res = variable.balance.quantile([lower_bound, upper_bound])
print(res)

max_threshold = variable['balance'].quantile(0.95)
max_threshold

variable[variable['balance']>max_threshold]

min_threshold = variable['balance'].quantile(0.05)
min_threshold

##Convert_string

variable.dtypes


from sklearn import preprocessing

def convert(variable):
    number = preprocessing.LabelEncoder()
    variable['ceducation'] = number.fit_transform(variable['education'])
    variable=variable.fillna(-999) # fill holes with default value
    return variable

import pandas as pd
Data = {'education', 'marital', 'age'}

variable = pd.DataFrame(Data)
print (variable)
print (variable.dtypes)	


import pandas as pd

Data = {'education': ['primary','secondary','tertiary'],
          'marital': ['divorced','married','single'],
       'housing': ['yes','no','yes'],
       'loan': ['yes','no','yes'],
       'class': ['no','yes','no']}

variable = pd.DataFrame(Data)
variable['marital'] = pd.to_numeric(variable['marital'])
variable['education'] = pd.to_numeric(variable['education'])
variable['housing'] = pd.to_numeric(variable['housing'])
variable['loan'] = pd.to_numeric(variable['loan'])
variable['class'] = pd.to_numeric(variable['class'])


print (variable)
print(variable.dtypes)

#one_Hot_Encoding

dummies = pd.get_dummies(variable.education)
print(dummies)
merged = pd.concat([variable,dummies], axis = 'columns')
merged

dummies = pd.get_dummies(variable.marital)
print(dummies)
merged = pd.concat([variable,dummies], axis = 'columns')
merged

final = merged.drop(['education', 'marital', 'campaign', 'pdays', 'previous'], axis='columns')
final

#Correlation_Coeff_Check

variable.cov()
variable.corr(method='pearson')

#Class_Distribution 

class_counts = variable.groupby('class').size()
print(class_counts)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)

pred_y=model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(test_y,pred_y))

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y,pred_y)


model.score(x_test, y_test)
variable['Target']=test_y.tolist()
variable['Predicted_Target'] = pred_y.tolist()


#Features_Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])


print(X_train)
print(X_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

print(y_train)
print(y_test)




