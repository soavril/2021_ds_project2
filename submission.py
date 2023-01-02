import numpy as np

predicted_intervals = [192, 465, 465, 604, 980, 1200, 1410, 1590, 1961, 2220, 2220, 2390, 2390, 2510, 2740, 2970, 3150, 3242, 3242, 3475, 3875, 4085, 5950, 7785, 8400, 8625, 9825, 10050, 10500, 10737, 10737, 10960, 10960, 11100, 12950, 13125, 13330, 13525, 13725, 13925, 15525, 15725, 15925, 16125, 16325, 16525]

rows_list = []

for i in range(50*round(predicted_intervals[0]) - 50):
  rows_list.append(0)

# print(len(rows_list))

for i in range(len(predicted_intervals)-1):
  start_time = predicted_intervals[i]
  end_time = predicted_intervals[i+1]
  # print(end_time- start_time)
  val = 0
  if i % 2 == 0:
    index = (i)//2
    val = pred_human[index]
    print(end_time, val)
    # print(val)
  for j in range(50*round(end_time - start_time)):
    rows_list.append(val)

  # print(len(rows_list))

# for j in range(50*round(predicted_intervals[33] - predicted_intervals[32])):
#   rows_list.append(predictions[16])

for i in range(841653-len(rows_list)):
  rows_list.append(0)  
submission = pd.DataFrame({'id':test.index, 'label':rows_list})

print(submission)

submission.to_csv('submission.csv',index=False)