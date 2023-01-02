import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from corr_filtering import *

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#cut first 2200s

plt.rcParams["figure.figsize"] = (18,5) 

def plot_time(df, time=(0, 100), columns='all'):
    df_time = df[(df['time']>=time[0]) & (df['time']<time[1])]
    column_list = df.columns if columns=='all' else columns

    for col in column_list:
        if col !='time':
            plt.plot('time', col, data=df_time)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()
    
plot_time(train, time=(20000, 35000), columns=['label', 's1'])
# plot_time(train, time=(1000, 10000), columns=['s1', 's2', 's5', 's10', 's12', 's16', 'label'])

plot_time(train, time=(1000, 100000), columns=['s2', 'label'])


#sliding window with 16.45*2 second

corr = train.corr(method='pearson')
array = []

array.append(corr.s1['label'])
array.append(corr.s2['label'])
array.append(corr.s3['label'])
array.append(corr.s4['label'])
array.append(corr.s5['label'])
array.append(corr.s6['label'])
array.append(corr.s7['label'])
array.append(corr.s8['label'])
array.append(corr.s9['label'])
array.append(corr.s10['label'])
array.append(corr.s11['label'])
array.append(corr.s12['label'])
array.append(corr.s13['label'])
array.append(corr.s14['label'])
array.append(corr.s15['label'])
array.append(corr.s16['label'])

print(array)

# s1: 0.65, s2: 0.038, s9: 0.38, s10: 0.43
# s3: 0.23, s4: 0.26, s11: 0.21, s12: 0.28
# s5: 0.53, s6: 0.60, s13: 0.48, s14: 0.50
# s7: 0.25, s8: 0.25, s15: 0.15, s16: 0.33

print(corr.describe)

#correlation filtering
cols_list = cor_filter(corr, threshold_y=0.4, threshold_x=0.4)
print(cols_list)


## check that label value changes every 1 second

for i in range(1, len(train['label'])):
  if train['label'][i-1] != train['label'][i]:
    if train['time'][i-1].is_integer() is False:
      print(train['time'][i-1])
      
      
#Group every 50 elements and apply Sliding Window

train_drop = train.drop(columns = {'id', 'time', 'label'})
test_drop = test.drop(columns = {'id', 'time'})
train_label = train['label']
y_train_shift = train['label']

# average every N number of  data

# hyperparameter: N = 50

N = 50 
train_temp = train_drop.groupby(np.arange(len(train_drop)) // N).mean()
test_temp = test_drop.groupby(np.arange(len(test_drop))  // N).mean()
y_train_temp = train_label.groupby(np.arange(len(train_label)) // N).mean()

#maximaze corr

N = 400
columns = ['s1', 's2','s3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
       's14', 's15', 's16']
for cols in columns:
  max_val = 0
  max_corr = 0
  for i in range(0, N):
    
    corr = train_temp[cols].shift(periods=-i).corr(y_train_temp[:-i])
    if corr > max_corr:
      max_corr = corr
      max_val = i
      # print(corr)
  if max_corr>0.0:
    print(cols, max_val, max_corr)
    
new_df = pd.DataFrame(y_train_temp[:-102])
new_df.reset_index(drop=True, inplace=True)

temp = train_temp['s1'][93:-9]
temp.reset_index(drop=True, inplace=True)

new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s2'][89:-13]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s3'][51:-51]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s4'][52:-50]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s5'][43:-59]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s6'][41:-61]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s7'][44:-58]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s8'][44:-58]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s9'][99:-3]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s10'][:-102]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s11'][55:-47]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s12'][52:-50]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s13'][46:-56]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s14'][45:-57]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s15'][44:-58]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)
temp = train_temp['s16'][44:-58]
temp.reset_index(drop=True, inplace=True)
new_df = pd.concat([new_df, temp], axis=1)

label2 = new_df['label']
train2 = new_df.drop(columns = 'label')

from sklearn.linear_model import LinearRegression
lm_model = LinearRegression()
lm_model.fit(train2, label2)
train_new = lm_model.predict(train2)
np.corrcoef(train_new, label2)

train_corr = pd.DataFrame(train_new)
train_corr[0][0] = 0
train_corr[0][1] = 0
train_corr[0][2] = 0
train_corr[0][3] = 0
train_corr

y_train_temp = new_df['label']
print(y_train_temp)

plt.rcParams["figure.figsize"] = (18,5) 

# len(train_temp['s1'])
time = range(len(train_corr[0]))

plt.plot(time, train_corr[0], color='b')
# plt.plot(time, train_temp['s10'].to_list(), color='g')
plt.plot(time, y_train_temp, color='r')

plt.xlabel('x axis')
plt.ylabel('y axis')

for xc in xcoords:
    plt.axvline(x=xc)

plt.show()

train_to_test = train_temp.tail(102)
train_to_test.reset_index(drop=True, inplace=True)
test1 = test_temp[102:]
test1.reset_index(drop=True, inplace=True)

test_temp = pd.concat([train_to_test, test1], axis = 0)
test_temp.reset_index(drop=True, inplace=True)
test_temp

test_temp_2 = lm_model.coef_ * test_temp
test_corr = pd.DataFrame(test_temp_2.sum(axis=1) + lm_model.intercept_)
test_corr

new_df['label'].corr(new_df['s1'])

train_temp = pd.DataFrame(new_df['s1'])
y_train_temp = pd.DataFrame(new_df['label'])

print(train_temp)
print(np.unique(train['label']))
print(y_train_temp)


