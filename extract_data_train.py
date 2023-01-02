import numpy as np

# Extract data from train set

final_X_train = []

for i in range(len(start_times)):
  start = start_times[i]
  end = end_times[i]
  
  temp = []  
  for cols in train_temp.columns:
    init = train_temp[cols][start]
    # max = init
    area = 0
    for j in range(start, end):
      # max variation
      cur = train_temp[cols][j]
      # if max < cur:
      #   max = cur
      # area
      area += cur
    # max var
    # temp.append(max-init)
    # area
    temp.append(area/(end-start))
    # duraiton
    # temp.append(end-start)
  if len(final_X_train) == 0:
    final_X_train = temp
  else:
    final_X_train = np.vstack([final_X_train, temp])
    
    
i = 2
label_lists = []

while i < len(new_label_dist):
  # print(i)
  if new_label_dist['label'][i] != 0:
    label_lists.append(new_label_dist['label'][i])
  i+= 1
  
print((label_lists))

test_final_x = pd.DataFrame(df_test[0])

# label_lists = (np.unique(train['label']))
print(len(np.unique(train['label'])))