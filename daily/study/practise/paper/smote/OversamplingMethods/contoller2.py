# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/18 16:40
# @Site    : 
# @File    : contoller2.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
from Standard_smote import *
from randomSmote import *
from AdaptiveOversampling import *
from BorderlineSmote import *
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import math
import random
from sklearn.metrics import confusion_matrix


blood_data = np.loadtxt(open("H:/Datamining/UCI/ionosphere/ionosphereData.csv", "rb"), delimiter=",", skiprows=0)
mine_lda_F_measure, mine_lda_G_mean = [], []
mine_lr_F_measure, mine_lr_G_mean = [], []
mine_nn_F_measure, mine_nn_G_mean = [], []
mine_svm_F_measure, mine_svm_G_mean = [], []

smOte_lda_F_measure, smOte_lda_G_mean = [], []
smOte_lr_F_measure, smOte_lr_G_mean = [], []
smOte_nn_F_measure, smOte_nn_G_mean = [], []
smOte_svm_F_measure, smOte_svm_G_mean = [], []

random_lda_F_measure, random_lda_G_mean = [], []
random_lr_F_measure, random_lr_G_mean = [], []
random_nn_F_measure, random_nn_G_mean = [], []
random_svm_F_measure, random_svm_G_mean = [], []

bor_lda_F_measure, bor_lda_G_mean = [], []
bor_lr_F_measure, bor_lr_G_mean = [], []
bor_nn_F_measure, bor_nn_G_mean = [], []
bor_svm_F_measure, bor_svm_G_mean = [], []

for time in range(20):
    train, test = cross_validation.train_test_split(blood_data, test_size=0.3)
    minority = []
    majority = []
    for i in train:
        if i[-1] == 1:
            minority.append(i[:-1].tolist())
        else:
            majority.append(i[:-1].tolist())
    ao = AdaptiveOversampling
    synthetic, new_minority, new_majority = ao(minority, majority, 20, 20).synthetic_generation()

    bor = BorderlineSmote
    bor_synthetic = bor(minority, majority, 5).over_sampling()

    smote = Smote
    n = int(float(len(majority))/float(len(minority)) * 100)
    smote_synthetic = smote(minority, n).over_sampling()

    random_smote = RandomSmote
    n = int(float(len(majority)) / float(len(minority)) * 100)
    random_smote_synthetic = random_smote(minority, n).over_sampling()

    for sy in synthetic:
        sy.append(1)
    for mi in new_minority:
        mi.append(1)
    for ma in new_majority:
        ma.append(-1)
    train = synthetic + new_majority + new_minority
    train_feature = []
    train_labels = []
    random.shuffle(train)
    for tr in train:
        train_feature.append(tr[:-1])
        train_labels.append(tr[-1])
    test_feature, test_labels = [], []
    for te in test:
        test_feature.append(te[:-1])
        test_labels.append(te[-1])

    new_smote_synthetic = []
    new_bor_synthetic = []
    new_ran_synthetic = []
    for bor in bor_synthetic:
        bor = bor.tolist()
        bor.append(1)
        new_bor_synthetic.append(bor)
    for syn in smote_synthetic:
        syn = syn.tolist()
        syn.append(1)
        new_smote_synthetic.append(syn)
    for r_syn in random_smote_synthetic:
        r_syn = r_syn.tolist()
        r_syn.append(1)
        new_ran_synthetic.append(r_syn)
    for mi in minority:
        mi.append(1)
    for ma in majority:
        ma.append(-1)
    smote_synthetic = new_smote_synthetic
    random_smote_synthetic = new_ran_synthetic
    bor_synthetic = new_bor_synthetic

    smote_train = smote_synthetic + minority + majority
    random.shuffle(smote_train)

    random_train = random_smote_synthetic + minority + majority
    random.shuffle(random_train)

    bor_train = bor_synthetic + minority + majority
    random.shuffle(bor_train)

    smote_train_feature = []
    smote_train_labels = []
    for tr in smote_train:
        smote_train_feature.append(tr[:-1])
        smote_train_labels.append(tr[-1])

    random_train_feature = []
    random_train_labels = []
    for tr in random_train:
        random_train_feature.append(tr[:-1])
        random_train_labels.append(tr[-1])

    bor_train_feature = []
    bor_train_labels = []
    for bor in bor_train:
        bor_train_feature.append(bor[:-1])
        bor_train_labels.append(bor[-1])
    # MINE
    lr = LogisticRegression()
    lr.fit(train_feature, train_labels)
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

    # SMOTE
    lr.fit(smote_train_feature, smote_train_labels)
    lr_test_predict = lr.predict(test_feature)
    smote_lr_tp, smote_lr_fn, smote_lr_fp, smote_lr_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lr_test_predict[i] and test_labels[i] == 1:
            smote_lr_tp += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == 1:
            smote_lr_fn += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == -1:
            smote_lr_fp += 1
        elif test_labels[i] == lr_test_predict[i] and test_labels[i] == -1:
            smote_lr_tn += 1

    # random smote
    lr.fit(random_train_feature, random_train_labels)
    lr_test_predict = lr.predict(test_feature)
    random_lr_tp, random_lr_fn, random_lr_fp, random_lr_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lr_test_predict[i] and test_labels[i] == 1:
            random_lr_tp += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == 1:
            random_lr_fn += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == -1:
            random_lr_fp += 1
        elif test_labels[i] == lr_test_predict[i] and test_labels[i] == -1:
            random_lr_tn += 1

    # bor_smote
    lr.fit(bor_train_feature, bor_train_labels)
    lr_test_predict = lr.predict(test_feature)
    bor_lr_tp, bor_lr_fn, bor_lr_fp, bor_lr_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lr_test_predict[i] and test_labels[i] == 1:
            bor_lr_tp += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == 1:
            bor_lr_fn += 1
        elif test_labels[i] != lr_test_predict[i] and test_labels[i] == -1:
            bor_lr_fp += 1
        elif test_labels[i] == lr_test_predict[i] and test_labels[i] == -1:
            bor_lr_tn += 1

    try:
        tpr = float(lr_tp) / (lr_tp + lr_fn)
        tnr = float(lr_tn) / (lr_tn + lr_fp)
        precision = float(lr_tp) / (lr_tp + lr_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        mine_lr_F_measure.append(F_measure)
        mine_lr_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass

    try:
        smote_tpr = float(smote_lr_tp) / (smote_lr_tp + smote_lr_fn)
        smote_tnr = float(smote_lr_tn) / (smote_lr_tn + smote_lr_fp)
        smote_precision = float(smote_lr_tp) / (smote_lr_tp + smote_lr_fp)
        smote_F_measure = float(2 * smote_tpr * smote_precision) / (smote_tpr + smote_precision)
        smote_G_mean = math.sqrt(smote_tpr * smote_tnr)
        smOte_lr_F_measure.append(smote_F_measure)
        smOte_lr_G_mean.append(smote_G_mean)
    except ZeroDivisionError:
        pass

    try:
        random_tpr = float(random_lr_tp) / (random_lr_tp + random_lr_fn)
        random_tnr = float(random_lr_tn) / (random_lr_tn + random_lr_fp)
        random_precision = float(random_lr_tp) / (random_lr_tp + random_lr_fp)
        random_F_measure = float(2 * random_tpr * random_precision) / (random_tpr + random_precision)
        random_G_mean = math.sqrt(random_tpr * random_tnr)
        random_lr_F_measure.append(random_F_measure)
        random_lr_G_mean.append(random_G_mean)
    except ZeroDivisionError:
        pass

    try:
        bor_tpr = float(bor_lr_tp) / (bor_lr_tp + bor_lr_fn)
        bor_tnr = float(bor_lr_tn) / (bor_lr_tn + bor_lr_fp)
        bor_precision = float(bor_lr_tp) / (bor_lr_tp + bor_lr_fp)
        bor_F_measure = float(2 * bor_tpr * bor_precision) / (bor_tpr + bor_precision)
        bor_G_mean = math.sqrt(bor_tpr * bor_tnr)
        bor_lr_F_measure.append(bor_F_measure)
        bor_lr_G_mean.append(bor_G_mean)
    except ZeroDivisionError:
        pass

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_feature, train_labels)
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
    try:
        tpr = float(lda_tp) / (lda_tp + lda_fn)
        tnr = float(lda_tn) / (lda_tn + lda_fp)
        precision = float(lda_tp) / (lda_tp + lda_fp)
        F_measure = float(2 * tpr * precision) / (tpr + precision)
        G_mean = math.sqrt(tpr * tnr)
        mine_lda_F_measure.append(F_measure)
        mine_lda_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass

    # SMOTE
    lda.fit(smote_train_feature, smote_train_labels)
    lda_test_predict = lda.predict(test_feature)
    smote_lda_tp, smote_lda_fn, smote_lda_fp, smote_lda_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lda_test_predict[i] and test_labels[i] == 1:
            smote_lda_tp += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == 1:
            smote_lda_fn += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == -1:
            smote_lda_fp += 1
        elif test_labels[i] == lda_test_predict[i] and test_labels[i] == -1:
            smote_lda_tn += 1
    try:
        smote_tpr = float(smote_lda_tp) / (smote_lda_tp + smote_lda_fn)
        smote_tnr = float(smote_lda_tn) / (smote_lda_tn + smote_lda_fp)
        smote_precision = float(smote_lda_tp) / (smote_lda_tp + smote_lda_fp)
        smote_F_measure = float(2 * smote_tpr * smote_precision) / (smote_tpr + smote_precision)
        smote_G_mean = math.sqrt(smote_tpr * smote_tnr)
        smOte_lda_F_measure.append(smote_F_measure)
        smOte_lda_G_mean.append(smote_G_mean)
    except ZeroDivisionError:
        pass

    lda.fit(random_train_feature, random_train_labels)
    lda_test_predict = lda.predict(test_feature)
    random_lda_tp, random_lda_fn, random_lda_fp, random_lda_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lda_test_predict[i] and test_labels[i] == 1:
            random_lda_tp += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == 1:
            random_lda_fn += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == -1:
            random_lda_fp += 1
        elif test_labels[i] == lda_test_predict[i] and test_labels[i] == -1:
            random_lda_tn += 1
    try:
        random_tpr = float(random_lda_tp) / (random_lda_tp + random_lda_fn)
        random_tnr = float(random_lda_tn) / (random_lda_tn + random_lda_fp)
        random_precision = float(random_lda_tp) / (random_lda_tp + random_lda_fp)
        random_F_measure = float(2 * random_tpr * random_precision) / (random_tpr + random_precision)
        random_G_mean = math.sqrt(random_tpr * random_tnr)
        random_lda_F_measure.append(random_F_measure)
        random_lda_G_mean.append(random_G_mean)
    except ZeroDivisionError:
        pass

    lda.fit(bor_train_feature, bor_train_labels)
    lda_test_predict = lda.predict(test_feature)
    bor_lda_tp, bor_lda_fn, bor_lda_fp, bor_lda_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == lda_test_predict[i] and test_labels[i] == 1:
            bor_lda_tp += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == 1:
            bor_lda_fn += 1
        elif test_labels[i] != lda_test_predict[i] and test_labels[i] == -1:
            bor_lda_fp += 1
        elif test_labels[i] == lda_test_predict[i] and test_labels[i] == -1:
            bor_lda_tn += 1
    try:
        bor_tpr = float(bor_lda_tp) / (bor_lda_tp + bor_lda_fn)
        bor_tnr = float(bor_lda_tn) / (bor_lda_tn + bor_lda_fp)
        bor_precision = float(bor_lda_tp) / (bor_lda_tp + bor_lda_fp)
        bor_F_measure = float(2 * bor_tpr * bor_precision) / (bor_tpr + bor_precision)
        bor_G_mean = math.sqrt(bor_tpr * bor_tnr)
        bor_lda_F_measure.append(bor_F_measure)
        bor_lda_G_mean.append(bor_G_mean)
    except ZeroDivisionError:
        pass

    # nn
    nn = MLPClassifier()
    nn.fit(train_feature, train_labels)
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
        mine_nn_F_measure.append(F_measure)
        mine_nn_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass

    nn.fit(smote_train_feature, smote_train_labels)
    nn_test_predict = nn.predict(test_feature)
    smote_nn_tp, smote_nn_fn, smote_nn_fp, smote_nn_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == nn_test_predict[i] and test_labels[i] == 1:
            smote_nn_tp += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == 1:
            smote_nn_fn += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == -1:
            smote_nn_fp += 1
        elif test_labels[i] == nn_test_predict[i] and test_labels[i] == -1:
            smote_nn_tn += 1
    # print "nn"
    # print nn_tp, nn_fn, nn_fp, nn_tn
    try:
        smote_tpr = float(smote_nn_tp) / (smote_nn_tp + smote_nn_fn)
        smote_tnr = float(smote_nn_tn) / (smote_nn_tn + smote_nn_fp)
        smote_precision = float(smote_nn_tp) / (smote_nn_tp + smote_nn_fp)
        smote_F_measure = float(2 * smote_tpr * smote_precision) / (smote_tpr + smote_precision)
        smote_G_mean = math.sqrt(smote_tpr * smote_tnr)
        smOte_nn_F_measure.append(smote_F_measure)
        smOte_nn_G_mean.append(smote_G_mean)
    except ZeroDivisionError:
        pass

    nn.fit(random_train_feature, random_train_labels)
    nn_test_predict = nn.predict(test_feature)
    random_nn_tp, random_nn_fn, random_nn_fp, random_nn_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == nn_test_predict[i] and test_labels[i] == 1:
            random_nn_tp += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == 1:
            random_nn_fn += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == -1:
            random_nn_fp += 1
        elif test_labels[i] == nn_test_predict[i] and test_labels[i] == -1:
            random_nn_tn += 1
    # print "nn"
    # print nn_tp, nn_fn, nn_fp, nn_tn
    try:
        random_tpr = float(random_nn_tp) / (random_nn_tp + random_nn_fn)
        random_tnr = float(random_nn_tn) / (random_nn_tn + random_nn_fp)
        random_precision = float(random_nn_tp) / (random_nn_tp + random_nn_fp)
        random_F_measure = float(2 * random_tpr * random_precision) / (random_tpr + random_precision)
        random_G_mean = math.sqrt(random_tpr * random_tnr)
        random_nn_F_measure.append(random_F_measure)
        random_nn_G_mean.append(random_G_mean)
    except ZeroDivisionError:
        pass

    nn.fit(bor_train_feature, bor_train_labels)
    nn_test_predict = nn.predict(test_feature)
    bor_nn_tp, bor_nn_fn, bor_nn_fp, bor_nn_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == nn_test_predict[i] and test_labels[i] == 1:
            bor_nn_tp += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == 1:
            bor_nn_fn += 1
        elif test_labels[i] != nn_test_predict[i] and test_labels[i] == -1:
            bor_nn_fp += 1
        elif test_labels[i] == nn_test_predict[i] and test_labels[i] == -1:
            bor_nn_tn += 1
    # print "nn"
    # print nn_tp, nn_fn, nn_fp, nn_tn
    try:
        bor__tpr = float(bor_nn_tp) / (bor_nn_tp + bor_nn_fn)
        bor_tnr = float(bor_nn_tn) / (bor_nn_tn + bor_nn_fp)
        bor_precision = float(bor_nn_tp) / (bor_nn_tp + bor_nn_fp)
        bor_F_measure = float(2 * bor_tpr * bor_precision) / (bor_tpr + bor_precision)
        bor_G_mean = math.sqrt(bor_tpr * bor_tnr)
        bor_nn_F_measure.append(bor_F_measure)
        bor_nn_G_mean.append(bor_G_mean)
    except ZeroDivisionError:
        pass

    # svm
    svm = LinearSVC()
    svm.fit(train_feature, train_labels)
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
        mine_svm_F_measure.append(F_measure)
        mine_svm_G_mean.append(G_mean)
    except ZeroDivisionError:
        pass

    svm.fit(smote_train_feature, smote_train_labels)
    svm_test_predict = svm.predict(test_feature)
    smote_svm_tp, smote_svm_fn, smote_svm_fp, smote_svm_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == svm_test_predict[i] and test_labels[i] == 1:
            smote_svm_tp += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == 1:
            smote_svm_fn += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == -1:
            smote_svm_fp += 1
        elif test_labels[i] == svm_test_predict[i] and test_labels[i] == -1:
            smote_svm_tn += 1
    # print "svm"
    # print svm_tp, svm_fn, svm_fp, svm_tn
    try:
        smote_tpr = float(smote_svm_tp) / (smote_svm_tp + smote_svm_fn)
        smote_tnr = float(smote_svm_tn) / (smote_svm_tn + smote_svm_fp)
        smote_precision = float(smote_svm_tp) / (smote_svm_tp + smote_svm_fp)
        smote_F_measure = float(2 * smote_tpr * smote_precision) / (smote_tpr + smote_precision)
        smote_G_mean = math.sqrt(smote_tpr * smote_tnr)
        smOte_svm_F_measure.append(smote_F_measure)
        smOte_svm_G_mean.append(smote_G_mean)
    except ZeroDivisionError:
        pass

    svm.fit(random_train_feature, random_train_labels)
    svm_test_predict = svm.predict(test_feature)
    random_svm_tp, random_svm_fn, random_svm_fp, random_svm_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == svm_test_predict[i] and test_labels[i] == 1:
            random_svm_tp += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == 1:
            random_svm_fn += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == -1:
            random_svm_fp += 1
        elif test_labels[i] == svm_test_predict[i] and test_labels[i] == -1:
            random_svm_tn += 1
    # print "svm"
    # print svm_tp, svm_fn, svm_fp, svm_tn
    try:
        random_tpr = float(random_svm_tp) / (random_svm_tp + random_svm_fn)
        random_tnr = float(random_svm_tn) / (random_svm_tn + random_svm_fp)
        random_precision = float(random_svm_tp) / (random_svm_tp + random_svm_fp)
        random_F_measure = float(2 * random_tpr * random_precision) / (random_tpr + random_precision)
        random_G_mean = math.sqrt(random_tpr * random_tnr)
        random_svm_F_measure.append(random_F_measure)
        random_svm_G_mean.append(random_G_mean)
    except ZeroDivisionError:
        pass

    svm.fit(bor_train_feature, bor_train_labels)
    svm_test_predict = svm.predict(test_feature)
    bor_svm_tp, bor_svm_fn, bor_svm_fp, bor_svm_tn = 0, 0, 0, 0
    for i in range(len(test_labels)):
        if test_labels[i] == svm_test_predict[i] and test_labels[i] == 1:
            bor_svm_tp += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == 1:
            bor_svm_fn += 1
        elif test_labels[i] != svm_test_predict[i] and test_labels[i] == -1:
            bor_svm_fp += 1
        elif test_labels[i] == svm_test_predict[i] and test_labels[i] == -1:
            bor_svm_tn += 1
    # print "svm"
    # print svm_tp, svm_fn, svm_fp, svm_tn
    try:
        bor_tpr = float(bor_svm_tp) / (bor_svm_tp + bor_svm_fn)
        bor_tnr = float(bor_svm_tn) / (bor_svm_tn + bor_svm_fp)
        bor_precision = float(bor_svm_tp) / (bor_svm_tp + bor_svm_fp)
        bor_F_measure = float(2 * bor_tpr * bor_precision) / (bor_tpr + bor_precision)
        bor_G_mean = math.sqrt(bor_tpr * bor_tnr)
        bor_svm_F_measure.append(bor_F_measure)
        bor_svm_G_mean.append(bor_G_mean)
    except ZeroDivisionError:
        pass

my_file = open('ionosphereData.txt', 'a')
my_file.write('SMOTE\n')
print "\n", "SMOTE"

print "lda"
my_file.write('lda:\n')
smote_lda_F_measure_avg = sum(smOte_lda_F_measure) / len(smOte_lda_F_measure)
smote_lda_F_measure_sd_sq = sum([(i - smote_lda_F_measure_avg) ** 2 for i in smOte_lda_F_measure])
smote_lda_F_measure_st_dev = (smote_lda_F_measure_sd_sq / (len(smOte_lda_F_measure) - 1)) ** .5
smote_lda_F_measure_avg = float('%.4f' % smote_lda_F_measure_avg)
smote_lda_F_measure_st_dev = float('%.4f' % smote_lda_F_measure_st_dev)
print "F_measure", smote_lda_F_measure_avg, "+_", smote_lda_F_measure_st_dev
my_file.write('F_measure:')
my_file.write(str(smote_lda_F_measure_avg))
my_file.write('+_')
my_file.write(str(smote_lda_F_measure_st_dev))
my_file.write('\n')
smote_lda_G_mean_avg = sum(smOte_lda_G_mean) / len(smOte_lda_G_mean)
smote_lda_G_mean_sd_sq = sum([(i - smote_lda_G_mean_avg) ** 2 for i in smOte_lda_G_mean])
smote_lda_G_mean_st_dev = (smote_lda_G_mean_sd_sq / (len(smOte_lda_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % smote_lda_G_mean_avg), "+_", float('%.4f' % smote_lda_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % smote_lda_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_lda_G_mean_st_dev)))
my_file.write('\n')

print "lr"
my_file.write('lr:\n')
smote_lr_F_measure_avg = sum(smOte_lr_F_measure) / len(smOte_lr_F_measure)
smote_lr_F_measure_sd_sq = sum([(i - smote_lr_F_measure_avg) ** 2 for i in smOte_lr_F_measure])
smote_lr_F_measure_st_dev = (smote_lr_F_measure_sd_sq / (len(smOte_lr_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % smote_lr_F_measure_avg), "+_", float('%.4f' % smote_lr_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % smote_lr_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_lr_F_measure_st_dev)))
my_file.write('\n')
smote_lr_G_mean_avg = sum(smOte_lr_G_mean) / len(smOte_lr_G_mean)
smote_lr_G_mean_sd_sq = sum([(i - smote_lr_G_mean_avg) ** 2 for i in smOte_lr_G_mean])
smote_lr_G_mean_st_dev = (smote_lr_G_mean_sd_sq / (len(smOte_lr_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % smote_lr_G_mean_avg), "+_", float('%.4f' % smote_lr_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % smote_lr_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_lr_G_mean_st_dev)))
my_file.write('\n')

print "nn"
my_file.write('nn:\n')
smote_nn_F_measure_avg = sum(smOte_nn_F_measure) / len(smOte_nn_F_measure)
smote_nn_F_measure_sd_sq = sum([(i - smote_nn_F_measure_avg) ** 2 for i in smOte_nn_F_measure])
smote_nn_F_measure_st_dev = (smote_nn_F_measure_sd_sq / (len(smOte_nn_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % smote_nn_F_measure_avg), "+_", float('%.4f' % smote_nn_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % smote_nn_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_nn_F_measure_st_dev)))
my_file.write('\n')
smote_nn_G_mean_avg = sum(smOte_nn_G_mean) / len(smOte_nn_G_mean)
smote_nn_G_mean_sd_sq = sum([(i - smote_nn_G_mean_avg) ** 2 for i in smOte_nn_G_mean])
smote_nn_G_mean_st_dev = (smote_nn_G_mean_sd_sq / (len(smOte_nn_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % smote_nn_G_mean_avg), "+_", float('%.4f' % smote_nn_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % smote_nn_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_nn_G_mean_st_dev)))
my_file.write('\n')

print "svm"
my_file.write('svm:\n')
smote_svm_F_measure_avg = sum(smOte_svm_F_measure) / len(smOte_svm_F_measure)
smote_svm_F_measure_sd_sq = sum([(i - smote_svm_F_measure_avg) ** 2 for i in smOte_svm_F_measure])
smote_svm_F_measure_st_dev = (smote_svm_F_measure_sd_sq / (len(smOte_svm_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % smote_svm_F_measure_avg), "+_", float('%.4f' % smote_svm_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % smote_svm_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_svm_F_measure_st_dev)))
my_file.write('\n')
smote_svm_G_mean_avg = sum(smOte_svm_G_mean) / len(smOte_svm_G_mean)
smote_svm_G_mean_sd_sq = sum([(i - smote_svm_G_mean_avg) ** 2 for i in smOte_svm_G_mean])
smote_svm_G_mean_st_dev = (smote_svm_G_mean_sd_sq / (len(smOte_svm_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % smote_svm_G_mean_avg), "+_", float('%.4f' % smote_svm_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % smote_svm_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % smote_svm_G_mean_st_dev)))
my_file.write('\n\n')

print "\n\n", "random"
my_file.write('random:\n')
print "lda"
my_file.write('lda:\n')
random_lda_F_measure_avg = sum(random_lda_F_measure) / len(random_lda_F_measure)
random_lda_F_measure_sd_sq = sum([(i - random_lda_F_measure_avg) ** 2 for i in random_lda_F_measure])
random_lda_F_measure_st_dev = (random_lda_F_measure_sd_sq / (len(random_lda_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % random_lda_F_measure_avg), "+_", float('%.4f' % random_lda_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % random_lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_lda_F_measure_st_dev)))
my_file.write('\n')
random_lda_G_mean_avg = sum(random_lda_G_mean) / len(random_lda_G_mean)
random_lda_G_mean_sd_sq = sum([(i - random_lda_G_mean_avg) ** 2 for i in random_lda_G_mean])
random_lda_G_mean_st_dev = (random_lda_G_mean_sd_sq / (len(random_lda_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % random_lda_G_mean_avg), "+_", float('%.4f' % random_lda_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % random_lda_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_lda_G_mean_st_dev)))
my_file.write('\n')

print "lr"
my_file.write('lr:\n')
random_lr_F_measure_avg = sum(random_lr_F_measure) / len(random_lr_F_measure)
random_lr_F_measure_sd_sq = sum([(i - random_lr_F_measure_avg) ** 2 for i in random_lr_F_measure])
random_lr_F_measure_st_dev = (random_lr_F_measure_sd_sq / (len(random_lr_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % random_lr_F_measure_avg), "+_", float('%.4f' % random_lr_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % random_lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_lda_F_measure_st_dev)))
my_file.write('\n')
random_lr_G_mean_avg = sum(random_lr_G_mean) / len(random_lr_G_mean)
random_lr_G_mean_sd_sq = sum([(i - random_lr_G_mean_avg) ** 2 for i in random_lr_G_mean])
random_lr_G_mean_st_dev = (random_lr_G_mean_sd_sq / (len(random_lr_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % random_lr_G_mean_avg), "+_", float('%.4f' % random_lr_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % random_lr_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_lr_G_mean_st_dev)))
my_file.write('\n')

print "nn"
my_file.write('nn:\n')
random_nn_F_measure_avg = sum(random_nn_F_measure) / len(random_nn_F_measure)
random_nn_F_measure_sd_sq = sum([(i - random_nn_F_measure_avg) ** 2 for i in random_nn_F_measure])
random_nn_F_measure_st_dev = (random_nn_F_measure_sd_sq / (len(random_nn_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % random_nn_F_measure_avg), "+_", float('%.4f' % random_nn_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % random_nn_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_nn_F_measure_st_dev)))
my_file.write('\n')
random_nn_G_mean_avg = sum(random_nn_G_mean) / len(random_nn_G_mean)
random_nn_G_mean_sd_sq = sum([(i - random_nn_G_mean_avg) ** 2 for i in random_nn_G_mean])
random_nn_G_mean_st_dev = (random_nn_G_mean_sd_sq / (len(random_nn_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % random_nn_G_mean_avg), "+_", float('%.4f' % random_nn_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % random_nn_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_nn_G_mean_st_dev)))
my_file.write('\n')

print "svm"
my_file.write('svm:\n')
random_svm_F_measure_avg = sum(random_svm_F_measure) / len(random_svm_F_measure)
random_svm_F_measure_sd_sq = sum([(i - random_svm_F_measure_avg) ** 2 for i in random_svm_F_measure])
random_svm_F_measure_st_dev = (random_svm_F_measure_sd_sq / (len(random_svm_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % random_svm_F_measure_avg), "+_", float('%.4f' % random_svm_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % random_svm_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_svm_F_measure_st_dev)))
my_file.write('\n')
random_svm_G_mean_avg = sum(random_svm_G_mean) / len(random_svm_G_mean)
random_svm_G_mean_sd_sq = sum([(i - random_svm_G_mean_avg) ** 2 for i in random_svm_G_mean])
random_svm_G_mean_st_dev = (random_svm_G_mean_sd_sq / (len(random_svm_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % random_svm_G_mean_avg), "+_", float('%.4f' % random_svm_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % random_svm_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % random_svm_G_mean_st_dev)))
my_file.write('\n\n')

print "\n\n", "bor"
my_file.write('bor:\n')
print "lda"
my_file.write('lda:\n')
lda_F_measure_avg = sum(bor_lda_F_measure) / len(bor_lda_F_measure)
lda_F_measure_sd_sq = sum([(i - lda_F_measure_avg) ** 2 for i in bor_lda_F_measure])
lda_F_measure_st_dev = (lda_F_measure_sd_sq / (len(bor_lda_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % lda_F_measure_avg), "+_", float('%.4f' % lda_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % lda_F_measure_st_dev)))
my_file.write('\n')
lda_G_mean_avg = sum(bor_lda_G_mean) / len(bor_lda_G_mean)
lda_G_mean_sd_sq = sum([(i - lda_G_mean_avg) ** 2 for i in bor_lda_G_mean])
lda_G_mean_st_dev = (lda_G_mean_sd_sq / (len(bor_lda_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % lda_G_mean_avg), "+_", float('%.4f' % lda_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % lda_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % lda_G_mean_st_dev)))
my_file.write('\n')


print "lr"
my_file.write('lr:\n')
lr_F_measure_avg = sum(bor_lr_F_measure) / len(bor_lr_F_measure)
lr_F_measure_sd_sq = sum([(i - lr_F_measure_avg) ** 2 for i in bor_lr_F_measure])
lr_F_measure_st_dev = (lr_F_measure_sd_sq / (len(bor_lr_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % lr_F_measure_avg), "+_", float('%.4f' % lr_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % lr_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % lr_F_measure_st_dev)))
my_file.write('\n')
lr_G_mean_avg = sum(bor_lr_G_mean) / len(bor_lr_G_mean)
lr_G_mean_sd_sq = sum([(i - lr_G_mean_avg) ** 2 for i in bor_lr_G_mean])
lr_G_mean_st_dev = (lr_G_mean_sd_sq / (len(bor_lr_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % lr_G_mean_avg), "+_", float('%.4f' % lr_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % lr_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % lr_G_mean_st_dev)))
my_file.write('\n')

print "nn"
my_file.write('nn:\n')
nn_F_measure_avg = sum(bor_nn_F_measure) / len(bor_nn_F_measure)
nn_F_measure_sd_sq = sum([(i - nn_F_measure_avg) ** 2 for i in bor_nn_F_measure])
nn_F_measure_st_dev = (nn_F_measure_sd_sq / (len(bor_nn_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % nn_F_measure_avg), "+_", float('%.4f' % nn_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % nn_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % nn_F_measure_st_dev)))
my_file.write('\n')
nn_G_mean_avg = sum(bor_nn_G_mean) / len(bor_nn_G_mean)
nn_G_mean_sd_sq = sum([(i - nn_G_mean_avg) ** 2 for i in bor_nn_G_mean])
nn_G_mean_st_dev = (nn_G_mean_sd_sq / (len(bor_nn_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % nn_G_mean_avg), "+_", float('%.4f' % nn_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % nn_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % nn_G_mean_st_dev)))
my_file.write('\n')

print "svm"
my_file.write('svm:\n')
svm_F_measure_avg = sum(bor_svm_F_measure) / len(bor_svm_F_measure)
svm_F_measure_sd_sq = sum([(i - svm_F_measure_avg) ** 2 for i in bor_svm_F_measure])
svm_F_measure_st_dev = (svm_F_measure_sd_sq / (len(bor_svm_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % svm_F_measure_avg), "+_", float('%.4f' % svm_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % svm_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % svm_F_measure_st_dev)))
my_file.write('\n')
svm_G_mean_avg = sum(bor_svm_G_mean) / len(bor_svm_G_mean)
svm_G_mean_sd_sq = sum([(i - svm_G_mean_avg) ** 2 for i in bor_svm_G_mean])
svm_G_mean_st_dev = (svm_G_mean_sd_sq / (len(bor_svm_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % svm_G_mean_avg), "+_", float('%.4f' % svm_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % svm_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % svm_G_mean_st_dev)))
my_file.write('\n\n')

print "\n", "MY_METHOD"

print "lda"
my_file.write('MY_METHOD:\n')
my_file.write('lda:\n')
mine_lda_F_measure_avg = sum(mine_lda_F_measure) / len(mine_lda_F_measure)
mine_lda_F_measure_sd_sq = sum([(i - mine_lda_F_measure_avg) ** 2 for i in mine_lda_F_measure])
mine_lda_F_measure_st_dev = (mine_lda_F_measure_sd_sq / (len(mine_lda_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % mine_lda_F_measure_avg), "+_", float('%.4f' % mine_lda_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % mine_lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lda_F_measure_st_dev)))
my_file.write('\n')
mine_lda_G_mean_avg = sum(mine_lda_G_mean) / len(mine_lda_G_mean)
mine_lda_G_mean_sd_sq = sum([(i - mine_lda_G_mean_avg) ** 2 for i in mine_lda_G_mean])
mine_lda_G_mean_st_dev = (mine_lda_G_mean_sd_sq / (len(mine_lda_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % mine_lda_G_mean_avg), "+_", float('%.4f' % mine_lda_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % mine_lda_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lda_G_mean_st_dev)))
my_file.write('\n')

print "lr"
my_file.write('lr:\n')
mine_lr_F_measure_avg = sum(mine_lr_F_measure) / len(mine_lr_F_measure)
mine_lr_F_measure_sd_sq = sum([(i - mine_lr_F_measure_avg) ** 2 for i in mine_lr_F_measure])
mine_lr_F_measure_st_dev = (mine_lr_F_measure_sd_sq / (len(mine_lr_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % mine_lr_F_measure_avg), "+_", float('%.4f' % mine_lr_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % mine_lr_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lr_F_measure_st_dev)))
my_file.write('\n')
mine_lr_G_mean_avg = sum(mine_lr_G_mean) / len(mine_lr_G_mean)
mine_lr_G_mean_sd_sq = sum([(i - mine_lr_G_mean_avg) ** 2 for i in mine_lr_G_mean])
mine_lr_G_mean_st_dev = (mine_lr_G_mean_sd_sq / (len(mine_lr_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % mine_lr_G_mean_avg), "+_", float('%.4f' % mine_lr_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % mine_lr_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lr_G_mean_st_dev)))
my_file.write('\n')

print "nn"
my_file.write('nn:\n')
mine_nn_F_measure_avg = sum(mine_nn_F_measure) / len(mine_nn_F_measure)
mine_nn_F_measure_sd_sq = sum([(i - mine_nn_F_measure_avg) ** 2 for i in mine_nn_F_measure])
mine_nn_F_measure_st_dev = (mine_nn_F_measure_sd_sq / (len(mine_nn_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % mine_nn_F_measure_avg), "+_", float('%.4f' % mine_nn_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % mine_lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lda_F_measure_st_dev)))
my_file.write('\n')
mine_nn_G_mean_avg = sum(mine_nn_G_mean) / len(mine_nn_G_mean)
mine_nn_G_mean_sd_sq = sum([(i - mine_nn_G_mean_avg) ** 2 for i in mine_nn_G_mean])
mine_nn_G_mean_st_dev = (mine_nn_G_mean_sd_sq / (len(mine_nn_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % mine_nn_G_mean_avg), "+_", float('%.4f' % mine_nn_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % mine_nn_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_nn_G_mean_st_dev)))
my_file.write('\n')

print "svm"
my_file.write('svm:\n')
mine_svm_F_measure_avg = sum(mine_svm_F_measure) / len(mine_svm_F_measure)
mine_svm_F_measure_sd_sq = sum([(i - mine_svm_F_measure_avg) ** 2 for i in mine_svm_F_measure])
mine_svm_F_measure_st_dev = (mine_svm_F_measure_sd_sq / (len(mine_svm_F_measure) - 1)) ** .5
print "F_measure", float('%.4f' % mine_svm_F_measure_avg), "+_", float('%.4f' % mine_svm_F_measure_st_dev)
my_file.write('F_measure:')
my_file.write(str(float('%.4f' % mine_lda_F_measure_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_lda_F_measure_st_dev)))
my_file.write('\n')
mine_svm_G_mean_avg = sum(mine_svm_G_mean) / len(mine_svm_G_mean)
mine_svm_G_mean_sd_sq = sum([(i - mine_svm_G_mean_avg) ** 2 for i in mine_svm_G_mean])
mine_svm_G_mean_st_dev = (mine_svm_G_mean_sd_sq / (len(mine_svm_G_mean) - 1)) ** .5
print "G_mean", float('%.4f' % mine_svm_G_mean_avg), "+_", float('%.4f' % mine_svm_G_mean_st_dev)
my_file.write('G_mean:')
my_file.write(str(float('%.4f' % mine_svm_G_mean_avg)))
my_file.write('+_')
my_file.write(str(float('%.4f' % mine_svm_G_mean_st_dev)))
my_file.write('\n')
my_file.close()
