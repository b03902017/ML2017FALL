import csv
import sys
import numpy as np
from numpy.linalg import inv # 用於跟 closed form 比較

# parameter seeting
h_look = 7 # 要看前幾個小時來預測下一小時
# features_list:
# [AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM2.5,
# RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR]
features = [9] # features we want to use
l_rate = 1
iterations = 20000
lamb = 0 # regularization param

# read training data
train_data = [[] for i in range(18)]
with open('../train.csv', encoding = 'big5') as f_train:
    for row_n, row in enumerate(csv.reader(f_train)):
        if row_n != 0: # row0 contains no data
             # row[3:]為 0~23 時
            row = list(map(lambda x: float(0) if (x == 'NR') else float(x), row[3:]))
            train_data[(row_n-1)%18].extend(row)

# read testing data
test_data = [[] for i in range(18)]
with open('../test.csv', encoding = 'big5') as f_test:
    for row_n, row in enumerate(csv.reader(f_test)):
        row = list(map(lambda x: float(0) if (x == 'NR') else float(x), row[2:]) )
        test_data[row_n%18].extend(row)
test_num = len(test_data[0]) // 9

# extract features
train_x = []
train_y = []
valid_x = []
valid_y = []
for month in range(12):
    for hour in range(480-h_look):
        x = [1]
        for i in features:
            x.extend(train_data[i][month*480+hour : month*480+hour+h_look])
            x.extend([j ** 2 for j in train_data[i][month*480+hour : month*480+hour+h_look]])
        if 480-h_look < hour:
            valid_x.append(x)
            valid_y.append(train_data[9][month*480+hour+h_look])
        elif hour < 480-h_look:
            train_x.append(x)
            train_y.append(train_data[9][month*480+hour+h_look])
for day in range(test_num):
    for hour in range(9-h_look):
        x = [1]
        for i in features:
            x.extend(test_data[i][day*9+hour : day*9+hour+h_look])
            x.extend([j ** 2 for j in test_data[i][day*9+hour : day*9+hour+h_look]])
        train_x.append(x)
        train_y.append(test_data[9][day*9+hour+h_look])

train_x = np.array(train_x)
train_y = np.array(train_y)
valid_x = np.array(valid_x)
valid_y = np.array(valid_y)
n, dim = train_x.shape
print(f'train shape: {train_x.shape}')
print(f'valid shape: {valid_x.shape}')

# training
w = np.zeros(dim)
train_x_t = train_x.transpose()
square_g = np.zeros(dim)
for i in range(iterations):
    y_err = np.dot(w, train_x_t) - train_y
    gra = 2 * np.dot(y_err, train_x) + 2 * lamb * np.concatenate((np.array([0]),w[1:]))
    square_g += gra ** 2
    w -= l_rate * gra / np.sqrt(square_g)
    if i % 5000 == 0:
        L = np.sqrt(np.sum(y_err ** 2) / n) # RMSE
        valid_L = 0 if len(valid_x) == 0 \
            else np.sqrt(np.sum((np.dot(w, valid_x.transpose()) - valid_y) ** 2) / len(valid_x))
        print ('iteration: %d | Train Loss: %.2f | Valid Loss: %.2f' % (i, L, valid_L))

np.save('../model/m4_7h_2w_l0.npy',w)

# 跟 closed form 比較
# per_w = np.matmul(np.matmul(inv(np.matmul(train_x_t,train_x)),train_x_t),train_y)
# y_err = np.dot(per_w, train_x_t) - train_y
# L = np.sqrt(np.sum(y_err ** 2) / n)
# valid_L = np.sqrt(np.sum((np.dot(per_w, valid_x.transpose()) - valid_y) ** 2) / len(valid_x))
# print ('Train Loss: %.1f | Valid Loss: %.1f' % (L, valid_L))
