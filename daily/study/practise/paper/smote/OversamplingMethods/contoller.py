# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/17 15:30
# @Site    : 
# @File    : contoller.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
from AdaptiveOversampling import *
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import math
import random
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation


blood_data = np.loadtxt(open("H:/Datamining/UCI/Blood/transfusionData.csv", "rb"), delimiter=",", skiprows=0)

# blood_data = np.loadtxt(open("H:/Datamining/UCI/Glass/glass.csv", "rb"), delimiter=",", skiprows=0)
F_measure_list = []
G_mean_list = []
kf = KFold(len(blood_data), n_folds=8)
for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print len(train_index), len(test_index)
    minority = []
    majority = []
    for i in train_index:
        if blood_data[i][-1] == 1:
            minority.append(blood_data[i][:-1].tolist())
        else:
            majority.append(blood_data[i][:-1].tolist())
    ao = AdaptiveOversampling
    synthetic, new_minority, new_majority = ao(minority, majority, 5, 18).synthetic_generation()
    print synthetic
    for sy in synthetic:
        sy.append(-1)
    for mi in new_minority:
        mi.append(-1)
    for ma in new_majority:
        ma.append(1)
    train = synthetic + new_majority + new_minority
    train_feature = []
    train_labels = []
    random.shuffle(train)
    for tr in train:
        train_feature.append(tr[:-1])
        train_labels.append(tr[-1])
    test_feature = []
    test_labels = []
    for j in test_index:
        test_feature.append(blood_data[j][:-1])
        test_labels.append(blood_data[j][-1])
    svc = LogisticRegression()
    svc.fit(train_feature, train_labels)
    test_predict = svc.predict(test_feature)
    try:
        t = confusion_matrix(test_labels, test_predict)
        tp = t[0][0]
        fn = t[0][1]
        fp = t[1][0]
        tn = t[1][1]
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (tn + fp)
        precision = float(tp) / (tp + fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        F_measure_list.append(F_measure)
        G_mean_list.append(G_mean)
    except ZeroDivisionError:
        pass
print F_measure_list
F_measure_avg = sum(F_measure_list) / len(F_measure_list)
sd_sq = sum([(i - F_measure_avg) ** 2 for i in F_measure_list])
stdev = (sd_sq / (len(F_measure_list) - 1)) ** .5
print "F_measure", F_measure_avg, "+_", stdev
