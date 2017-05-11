# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/20 14:34
# @Site    : 
# @File    : other_smote.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

from AdaptiveOversampling import *
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import math
import random
from sklearn.metrics import confusion_matrix

trainDataSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                 "MATLAB-Oversampling-Methods/trainDataSMOTE.csv", "rb"), delimiter=",", skiprows=0)

trainLabelSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                  "MATLAB-Oversampling-Methods/trainLabelSMOTE.csv", "rb"), delimiter=",", skiprows=0)

trainDatanewBorSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                       "MATLAB-Oversampling-Methods/trainDatanewBorSMOTE.csv", "rb"), delimiter=",", skiprows=0)

trainLabelnewBorSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                        "MATLAB-Oversampling-Methods/trainLabelnewBorSMOTE.csv", "rb"), delimiter=",", skiprows=0)


trainDatanewSafeSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                        "MATLAB-Oversampling-Methods/trainDatanewSafeSMOTE.csv", "rb"), delimiter=",", skiprows=0)

trainLabelnewSafeSMOTE = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/"
                                         "MATLAB-Oversampling-Methods/trainLabelnewSafeSMOTE.csv", "rb"), delimiter=",", skiprows=0)

test_feature = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/MATLAB-Oversampling-Methods/"
                           "testData.csv", "rb"), delimiter=",", skiprows=0)

test_labels = np.loadtxt(open("C:/Users/zhouxinmin/PycharmProjects/study/practise/paper/MATLAB-Oversampling-Methods/"
                            "testLabel.csv", "rb"), delimiter=",", skiprows=0)

lda_F_measure, lda_G_mean = [], []
lr_F_measure, lr_G_mean = [], []
nn_F_measure, nn_G_mean = [], []
svm_F_measure, svm_G_mean = [], []

for time in range(20):
    lr = LogisticRegression()
    lr.fit(trainDataSMOTE, trainLabelSMOTE)
    lr_test_predict = lr.predict(test_feature)
    lr_tp, lr_fn, lr_fp, lr_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lr_test_predict[i] and test_labels[i] == 1:
            lr_tp += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == 1:
            lr_fn += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == -1:
            lr_fp += 1
        elif test_labels[i] == lr_test_predict[i] and test_labels[i] == -1:
            lr_tn += 1
    # print "LogisticRegression"
    # print lr_tp, lr_fn, lr_fp, lr_tn
    try:
        tpr = float(lr_tp) / (lr_tp + lr_fn)
        tnr = float(lr_tn) / (lr_tn + lr_fp)
        precision = float(lr_tp) / (lr_tp + lr_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        lr_F_measure.append(F_measure)
        lr_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass
    # print 'F_measure:', F_measure, 'G_mean', G_mean

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainDataSMOTE, trainLabelSMOTE)
    lda_test_predict = lda.predict(test_feature)
    lda_tp, lda_fn, lda_fp, lda_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lda_test_predict[i] and test_labels[i] == 1:
            lda_tp += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == 1:
            lda_fn += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == -1:
            lda_fp += 1
        elif test_labels[i] == lda_test_predict[i] and test_labels[i] == -1:
            lda_tn += 1
    # print "LDA"
    # print lda_tp, lda_fn, lda_fp, lda_tn
    try:
        tpr = float(lda_tp) / (lda_tp + lda_fn)
        tnr = float(lda_tn) / (lda_tn + lda_fp)
        precision = float(lda_tp) / (lda_tp + lda_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        lda_F_measure.append(F_measure)
        lda_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass
    # print 'F_measure:', F_measure, 'G_mean', G_mean

    nn = MLPClassifier()
    nn.fit(trainDataSMOTE, trainLabelSMOTE)
    nn_test_predict = nn.predict(test_feature)
    nn_tp, nn_fn, nn_fp, nn_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == nn_test_predict[i] and test_labels[i] == 1:
            nn_tp += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == 1:
            nn_fn += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == -1:
            nn_fp += 1
        elif test_labels[i] == nn_test_predict[i] and test_labels[i] == -1:
            nn_tn += 1
    # print "nn"
    # print nn_tp, nn_fn, nn_fp, nn_tn
    try:
        tpr = float(nn_tp) / (nn_tp + nn_fn)
        tnr = float(nn_tn) / (nn_tn + nn_fp)
        precision = float(nn_tp) / (nn_tp + nn_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        nn_F_measure.append(F_measure)
        nn_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass
    # print 'F_measure:', F_measure, 'G_mean', G_mean

    svm = LinearSVC()
    svm.fit(trainDataSMOTE, trainLabelSMOTE)
    svm_test_predict = svm.predict(test_feature)
    svm_tp, svm_fn, svm_fp, svm_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == svm_test_predict[i] and test_labels[i] == 1:
            svm_tp += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == 1:
            svm_fn += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == -1:
            svm_fp += 1
        elif test_labels[i] == svm_test_predict[i] and test_labels[i] == -1:
            svm_tn += 1
    # print "svm"
    # print svm_tp, svm_fn, svm_fp, svm_tn
    try:
        tpr = float(svm_tp) / (svm_tp + svm_fn)
        tnr = float(svm_tn) / (svm_tn + svm_fp)
        precision = float(svm_tp) / (svm_tp + svm_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        svm_F_measure.append(F_measure)
        svm_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass
        # print 'F_measure:', F_measure, 'G_mean', G_mean
print "lda"
lda_F_measure_avg = sum(lda_F_measure) / len(lda_F_measure)
lda_F_measure_sd_sq = sum([(i - lda_F_measure_avg) ** 2 for i in lda_F_measure])
lda_F_measure_st_dev = (lda_F_measure_sd_sq / (len(lda_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % lda_F_measure_avg), "+_", float('%.4f' % lda_F_measure_st_dev)
lda_G_mean_avg = sum(lda_G_mean) / len(lda_G_mean)
lda_G_mean_sd_sq = sum([(i - lda_G_mean_avg) ** 2 for i in lda_G_mean])
lda_G_mean_st_dev = (lda_G_mean_sd_sq / (len(lda_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % lda_G_mean_avg), "+_", float('%.4f' % lda_G_mean_st_dev)

print "lr"
lr_F_measure_avg = sum(lr_F_measure) / len(lr_F_measure)
lr_F_measure_sd_sq = sum([(i - lr_F_measure_avg) ** 2 for i in lr_F_measure])
lr_F_measure_st_dev = (lr_F_measure_sd_sq / (len(lr_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % lr_F_measure_avg), "+_", float('%.4f' % lr_F_measure_st_dev)
lr_G_mean_avg = sum(lr_G_mean) / len(lr_G_mean)
lr_G_mean_sd_sq = sum([(i - lr_G_mean_avg) ** 2 for i in lr_G_mean])
lr_G_mean_st_dev = (lr_G_mean_sd_sq / (len(lr_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % lr_G_mean_avg), "+_", float('%.4f' % lr_G_mean_st_dev)

print "nn"
nn_F_measure_avg = sum(nn_F_measure) / len(nn_F_measure)
nn_F_measure_sd_sq = sum([(i - nn_F_measure_avg) ** 2 for i in nn_F_measure])
nn_F_measure_st_dev = (nn_F_measure_sd_sq / (len(nn_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % nn_F_measure_avg), "+_", float('%.4f' % nn_F_measure_st_dev)
nn_G_mean_avg = sum(nn_G_mean) / len(nn_G_mean)
nn_G_mean_sd_sq = sum([(i - nn_G_mean_avg) ** 2 for i in nn_G_mean])
nn_G_mean_st_dev = (nn_G_mean_sd_sq / (len(nn_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % nn_G_mean_avg), "+_", float('%.4f' % nn_G_mean_st_dev)

print "svm"
svm_F_measure_avg = sum(svm_F_measure) / len(svm_F_measure)
svm_F_measure_sd_sq = sum([(i - svm_F_measure_avg) ** 2 for i in svm_F_measure])
svm_F_measure_st_dev = (svm_F_measure_sd_sq / (len(svm_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % svm_F_measure_avg), "+_", float('%.4f' % svm_F_measure_st_dev)
svm_G_mean_avg = sum(svm_G_mean) / len(svm_G_mean)
svm_G_mean_sd_sq = sum([(i - svm_G_mean_avg) ** 2 for i in svm_G_mean])
svm_G_mean_st_dev = (svm_G_mean_sd_sq / (len(svm_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % svm_G_mean_avg), "+_", float('%.4f' % svm_G_mean_st_dev)