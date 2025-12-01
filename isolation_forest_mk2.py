
#isolation Forest tinkering
#https://www.digitalocean.com/community/tutorials/anomaly-detection-isolation-forest


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


os.chdir('/Users/mike/Documents/python_stuff/Isolation_Forest_Mk2')
cwd = os.getcwd()
print(cwd)


df = pd.read_csv("salary2.csv")
df.head(10)


model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['Salary']])


df['scores']=model.decision_function(df[['Salary']])
df['anomaly']=model.predict(df[['Salary']])
df.head(20)


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


outliers_counter = len(df[df['Salary'] > 99999])
outliers_counter


print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))