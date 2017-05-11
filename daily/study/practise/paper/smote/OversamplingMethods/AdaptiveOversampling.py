# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/13 15:33
# @Site    : 
# @File    : AdaptiveOversampling.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import numpy as np
import copy
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans


class AdaptiveOversampling:
    def __init__(self, minority, majority, k, n_cluster):
        self.minority = copy.deepcopy(minority)
        self.majority = copy.deepcopy(majority)
        train = np.row_stack((self.minority, self.majority))
        self.train = train
        self.k = k
        self.n_cluster = n_cluster
        self.new_index = 0
        self.synthetic = []

    def noise_remover(self):
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.train)
        noise_minority = []
        for a_min in range(len(self.minority)-1):
            n_array = neighbors.kneighbors(self.minority[a_min], return_distance=False)[0]
            count = 0
            for bor in n_array.tolist():
                ones = self.train[bor].tolist()
                if ones in self.majority:
                    count += 1
            if count == len(n_array):
                noise_minority.append(self.minority[a_min])
        for i in noise_minority:
            self.minority.remove(i)

        noise_majority = []
        for a_maj in range(len(self.majority)):
            n_array = neighbors.kneighbors(self.majority[a_maj], return_distance=False)[0]
            count = 0
            for bor in n_array:
                ones = self.train[bor].tolist()
                if ones in self.minority:
                    count += 1
            if count == len(n_array):
                noise_majority.append(self.majority[a_maj])
        for j in noise_majority:
            self.majority.remove(j)

    def clustering(self):
        self.noise_remover()
        minority_k_means = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np.array(self.minority))
        minority_n_labels = minority_k_means.labels_
        minority_n_labels_set = copy.deepcopy(minority_n_labels)
        minority_n_labels_set = set(minority_n_labels_set)
        minority_n_dict = {}
        for i in minority_n_labels_set:
            minority_n_dict[i] = []
        for index, value in enumerate(minority_n_labels):
            minority_n_dict[value].append(index)
        err_rate_dic = {}
        err_sum = 0
        for labels in minority_n_labels_set:
            test = []
            for t in minority_n_dict[labels]:
                test.append(self.minority[t])
            train = copy.deepcopy(self.minority)
            for be_test in test:
                train.remove(be_test)
            train_feature = train + self.majority
            train_labels = [-1] * len(train) + [1] * len(self.majority)
            lda = LinearDiscriminantAnalysis()
            lda.fit(train_feature, train_labels)
            predict_labels = lda.predict(test).tolist()
            err = predict_labels.count(1)
            err_rate = float(err)/len(predict_labels)
            err_sum += err_rate
            # err_rate_list.append(err_rate)
            err_rate_dic[labels] = err_rate
        # print err_sum
        for key in err_rate_dic:
            err_rate_dic[key] /= err_sum
        return minority_n_dict, err_rate_dic

    def synthetic_generation(self):
        n_cluser, err_dic = self.clustering()
        margin_instance = int(len(self.majority)/2 - len(self.minority)/2)
        # margin_instance = len(self.majority) - len(self.minority)
        for key, err_value in err_dic.items():
            err_dic[key] = int(err_value * margin_instance)
        for key, value in n_cluser.items():
            n = err_dic[key]
            samples = n_cluser[key]
            self.smote(n, samples)
        return self.synthetic, self.minority, self.majority

    def smote(self, n, samples):
        for i in range(n):
            index_list = [random.randint(0, len(samples)) for _ in range(2)]
            dif = np.array(self.minority[index_list[0]]) - np.array(self.minority[index_list[1]])
            gap = np.random.rand(1, len(self.minority[0]))
            # self.synthetic[self.new_index] = self.minority[index_list[1]] + gap.flatten() * dif
            new_synthetic = (self.minority[index_list[1]] + gap.flatten() * dif).tolist()
            self.synthetic.append(new_synthetic)
