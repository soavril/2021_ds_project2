import numpy as np

label_dist = y_train_shift

data_list = []

prev_value = label_dist[0]
current_value = label_dist[0]
current_count = 0

# hyperparameter: label_dist[i] > 100
# hyperparameter: i - current_count >= 50

for i in range(len(label_dist)):
  # print(i)
  if (current_value == 0 or label_dist[i] == 0) and current_value != label_dist[i]:
    print(i)
    # check the change lasted sufficiently long time
    # append information
    data_list.append({"start_time": current_count, "end_time": i, "label": current_value, "prev_label": prev_value})

    # update for the next round
    prev_value = current_value
    current_value = label_dist[i]
    current_count = i

data_list.append({"start_time": current_count, "end_time": i, "label": current_value, "prev_label": prev_value})
new_label_dist = pd.DataFrame(data_list)

print(new_label_dist)

for i in range(len(new_label_dist)):
  new_label_dist['start_time'][i] = round(new_label_dist['start_time'][i]/50)
  new_label_dist['end_time'][i] = round(new_label_dist['end_time'][i]/50)  
  
print(new_label_dist)

real_list = []
for i in range(2, len(new_label_dist)):
  if new_label_dist['label'][i] != 0.00:
    real_list.append(new_label_dist['start_time'][i])
    real_list.append(new_label_dist['end_time'][i])
    
print(len(real_list))

#regresiion of intervals

print(len(real_list))
print(len(xcoords))

from sklearn.linear_model import LinearRegression
lm_model = LinearRegression()
X = np.array(xcoords)
lm_model.fit(X.reshape(-1,1), real_list)
# from sklearn import linear_model
# lasso = linear_model.Lasso(alpha = 3)
# lasso.fit(X.reshape(-1,1), real_list)

print(lm_model.coef_)

pred_train = lm_model.predict(X.reshape(-1,1))

X_pred = np.array(xcoords_test)
predicted_intervals = lm_model.predict(X_pred.reshape(-1,1))

X_pred_temp = np.array(xcoords_test)
predicted_intervals_temp = lasso.predict(X_pred_temp.reshape(-1,1))