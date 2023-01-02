# threshold_y: threshold for correlation with label
# threshold_x: threshold for correlation with other columns
# max_cols: number of maximum columns to choose from
# corr: correlation array

# utility function used in cor_filter; selects the best column based on corr_y
def add_column_y(corr, cols_list, fixed_list, threshold_y):
  max_corr = -1
  prev_column = "null"
  for col in cols_list:
    if col not in fixed_list:
      if corr[col]['label'] > max_corr and corr[col]['label'] > threshold_y:
        max_corr = corr[col]['label']
        prev_column = col 
  return prev_column

# utility function used in cor_filter; selects the best column based on corr_x
def remove_column_x(corr, cols_list, fixed_list, next_column, threshold_x):
  for col in cols_list:
    if col not in fixed_list:
      if corr[next_column][col] < threshold_x:
        cols_list.remove(col)

def cor_filter(corr, threshold_y=0.4, threshold_x=0.4, max_cols=-1):
  cols_list=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
       's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']
  fixed_list = []
  while len(cols_list) > max_cols:
    next_column = add_column_y(corr, cols_list, fixed_list, threshold_y)
    if next_column == "null":
      break
    else:
      fixed_list.append(next_column)
    remove_column_x(corr, cols_list, fixed_list, next_column, threshold_x)

  return fixed_list