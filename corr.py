import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

x = train.drop(columns = {'id','time', 'label'})
x1 = x[['s1','s2','s9','s10']]
x2 = x[['s3','s4','s11','s12']]
x3 = x[['s5','s6','s13','s14']]
x4 = x[['s7','s8','s15','s16']]

#corr between group
plt.figure(figsize=(10,10))
plt.subplot(221)
sns.heatmap(data = x1.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
plt.subplot(222)
sns.heatmap(data = x2.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
plt.subplot(223)
sns.heatmap(data = x3.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
plt.subplot(224)
sns.heatmap(data = x4.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')

#corr between all
plt.figure(figsize=(10,10))
sns.heatmap(data = x.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')