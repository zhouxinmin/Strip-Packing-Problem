# coding:utf-8

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy
import math
from sklearn.metrics import roc_curve, auc

# germandata = numpy.loadtxt(open("H:/Datamining/UCI/Glass/SMOTE_glass.csv", "rb"), delimiter=","
#                            , skiprows=0)
# train = germandata[:191]    # train
# test = germandata[191:]     # test

germandata = numpy.loadtxt(open("H:/Datamining/UCI/Iris/bezdekIris.csv", "rb"), delimiter=","
                           , skiprows=0)
train = germandata[:100]    # train
test = germandata[100:]     # test

# germandata = numpy.loadtxt(open("H:/Datamining/UCI/Glass/ESMOTE_glass.csv", "rb"), delimiter=","
#                            , skiprows=0)
# train = germandata[:258]    # train
# test = germandata[258:]     # test

train_x = train[:, 0:-1]
train_y = train[:, -1]
test_x = test[:, 0:-1]
test_y = test[:, -1]

clf = svm.SVC()
clf.fit(train_x, train_y)
svm_predict = clf.predict(test_x)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(test_y, svm_predict)
roc_auc = auc(svm_fpr, svm_tpr)
print svm_predict
print 'svm'
t = confusion_matrix(test_y, svm_predict)
print t
tp = t[0][0]
fn = t[0][1]
fp = t[1][0]
tn = t[1][1]
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
precision = float(tp)/(tp+fp)
F_measure = float(2*tpr*precision)/(tpr+precision)
G_mean = math.sqrt(tpr*tnr)
print 'F_measure:', F_measure, 'G_mean', G_mean

# nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)
# nn.fit(train_x, train_y)
# nn_predict = nn.predict(test_x)
# nn_fpr, nn_tpr, nn_thresholds = roc_curve(test_y, nn_predict)
# nn_roc_auc = auc(nn_fpr, nn_tpr)
# print 'classification'
# n = confusion_matrix(test_y, nn_predict)
# tp = n[0][0]
# fn = n[0][1]
# fp = n[1][0]
# tn = n[1][1]
# tpr = float(tp)/(tp+fn)
# tnr = float(tn)/(tn+fp)
# precision = float(tp)/(tp+fp)
# F_measure = float(2*tpr*precision)/(tpr+precision)
# G_mean = math.sqrt(tpr*tnr)
# print F_measure, G_mean

random = RandomForestClassifier(random_state=123)
random.fit(train_x, train_y)
rf_pred = random.predict(test_x)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(test_y, rf_pred)
rf_roc_auc = auc(rf_fpr, rf_tpr)
print 'randomf'
# print classification_report(test_y, rf_pred)
m = confusion_matrix(test_y, rf_pred)
# print m
tp = m[0][0]
fn = m[0][1]
fp = m[1][0]
tn = m[1][1]
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
precision = float(tp)/(tp+fp)
F_measure = float(2*tpr*precision)/(tpr+precision)
G_mean = math.sqrt(tpr*tnr)
print 'F_measure:', F_measure, 'G_mean', G_mean

# 这里用Logistic回归
lr_model = LogisticRegression(C=1.0, penalty='l2')
lr_model.fit(train_x, train_y)
lr_pred = lr_model.predict(test_x)
lr_fpr, lr_tpr, lr_thresholds = roc_curve(test_y, lr_pred)
lr_roc_auc = auc(lr_fpr, lr_tpr)
print 'logistic'
h = confusion_matrix(test_y, lr_pred)
# print h
tp = h[0][0]
fn = h[0][1]
fp = h[1][0]
tn = h[1][1]
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
precision = float(tp)/(tp+fp)
F_measure = float(2*tpr*precision)/(tpr+precision)
G_mean = math.sqrt(tpr*tnr)
print 'F_measure:', F_measure, 'G_mean', G_mean