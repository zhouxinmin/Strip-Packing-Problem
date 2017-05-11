# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/5/3 11:01
# @Site    : 
# @File    : test.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import numpy as np
from sklearn import preprocessing
# from sklearn.utils import shuffle


def pickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

credit_card = np.loadtxt(open("credit_card_clients_import.csv", "rb"), delimiter=",", skiprows=0)

data_set_x = credit_card[:, :-1]
data_set_y = credit_card[:, -1]
min_max_scaler = preprocessing.MinMaxScaler()
data_set_x_minmax = min_max_scaler.fit_transform(data_set_x)
new_data_set_x = []
for x in data_set_x_minmax:
    a = x.reshape(4, 6)
    b = a.T
    new_data_set_x.append(b)
print("shuffle data...")
# data_set_x, data_set_y = shuffle(data_set_x, data_set_y, random_state=0)
print("shuffle data done !")
X_train, X_val, X_test = new_data_set_x[:19200], new_data_set_x[19200:24000], new_data_set_x[24000:]
y_train, y_val, y_test = data_set_y[:19200], data_set_y[19200:24000], data_set_x[24000:]
print len(X_train), len(X_val), len(X_test)
print len(y_train), len(y_val), len(y_test)
print X_train[:7]
print len(data_set_y)
# print credit_card
# print type(credit_card)
# pickle_output((X_train, y_train), "train")
# pickle_output((X_val, y_val), "val")
# pickle_output((X_test, y_test), "test")
