# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

train.info()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

# distribution of labels in training data
sns.distplot(train.label)
plt.xticks(rotation=45)
plt.show()

# unique values in label coulum
np.unique(train.label)

sns.countplot(train.label)
plt.xticks(rotation=45)
plt.show()


# What percentage of labels in training set equal to 0?
train[train.label == 0].shape[0] / train.shape[0]

# scatterplot + histogram of time and label
sns.jointplot(x='time', y='label', data=train)
plt.show()

# scatterplot + histogram of s1 and label
sns.jointplot(x='s1', y='label', data=train)
plt.show()


from sklearn.linear_model import LinearRegression

X = train.copy()
x_cols = ['s'+ str(i) for i in list(range(1,17,1))]
X = X[x_cols]
X.head()

y = train['label']

lm_model = LinearRegression()
lm_model.fit(X, y)

new_X = test[x_cols]
new_y = lm_model.predict(new_X)

submission_lm = submission.copy()
submission_lm['label'] = new_y

submission_lm.to_csv('submission_lm.csv', index=False)

corr = train.corr(method='pearson')
print(corr.s9)