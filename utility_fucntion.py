import numpy as np
import pandas as np

threshold_start = 200
threshold_end = 200
# start and end times of non-zero labels

start_times = []
end_times = []
xcoords = []

threshold_value = 240

i = 0
while i < len(train_corr[0]):
  val = train_corr[0][i]
  if val >= threshold_start:
    save = i
    flag = False
    threshold = threshold_start
    while train_corr[0][i] >= threshold:
      if train_corr[0][i] >= threshold_value and flag is False:
        if save > 2000:
          start_times.append(save)
          xcoords.append(save)
          flag = True
        threshold = threshold_end
      i += 1
      if i == len(train_corr[0]):
        break
    if flag is True:
      if i > 2000:
        end_times.append(i)
        xcoords.append(i)
  i += 1
  
print(len(start_times))
print(len(end_times))
print(start_times)
print(end_times)

print(xcoords_test)

#verify with testdata

# start and end times of non-zero labels
# dictionary = [{"start_times": 167, "end_times": 366}]

start_times_test = []
end_times_test = []
xcoords_test = []

threshold_value = 260
i = 0

while i < len(test_corr):
  val = test_corr[0][i]
  if val >= threshold_start:
    save = i
    flag = False
    threshold = threshold_start
    while test_corr[0][i] >= threshold:
      if test_corr[0][i] >= threshold_value and flag is False:
        start_times_test.append(save)
        xcoords_test.append(save)
        flag = True
        threshold = threshold_end
      i += 1
      if i == len(test_corr):
        break
    if flag is True:
      end_times_test.append(i)
      xcoords_test.append(i)
  i += 1
  
print(len(start_times_test))
print(len(end_times_test))
print(start_times_test)
print(end_times_test)

print(xcoords_test)