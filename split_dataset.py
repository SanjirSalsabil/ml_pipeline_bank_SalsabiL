##split dataset

import pandas as pd
import pandas as pd
import os
import re
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv(r"/Users/sanjirsalsabil/Desktop/bank.csv")
df.head()
df_train,df_test = train_test_split(df, test_size=0.35, random_state=100)

df_train.to_csv("bank_training.csv",index=False)
df_test.to_csv("bank_testing.csv",index=False)

df_train.nunique() 

df_train.info()

set_option('display.width', 100)
set_option('precision', 3)
correlations = df.corr(method='pearson')
print(correlations)
