import csv
import sys
import numpy as np

# parameter seeting
h_look = 7 # 要看前幾個小時來預測下一小時
features = [9]

# read testing data
test_data = [[] for i in range(18)]
with open(sys.argv[1], encoding = 'big5') as f_test:
    for row_n, row in enumerate(csv.reader(f_test)):
        row = list(map(lambda x: float(0) if (x == 'NR') else float(x), row[2:]) )
        test_data[row_n%18].extend(row)
test_num = len(test_data[0]) // 9

# testing
w = np.load('./model/best_model.npy')
test_x = []
for day in range(test_num):
    x = [1]
    for i in features:
        x.extend(test_data[i][(day+1)*9-h_look : (day+1)*9])
        x.extend([j ** 2 for j in test_data[i][(day+1)*9-h_look : (day+1)*9]])
    test_x.append(x)
test_x = np.array(test_x)
ans = np.dot(w, test_x.transpose())
with open(sys.argv[2], 'w') as f_out:
    print('id,value', file=f_out)
    for i in range(len(ans)):
        print(f'id_{i},{ans[i]}', file=f_out)
