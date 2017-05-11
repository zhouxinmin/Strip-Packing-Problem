# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/6 20:13
# @Site    : 
# @File    : A-SUWO.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import copy


class AdaptiveOversampling:
    def __init__(self, minority, majority, k, n_cluster):
        self.minority = copy.deepcopy(minority)
        self.majority = copy.deepcopy(majority)
        train = np.row_stack((self.minority, self.majority))
        self.train = train.tolist()
        self.k = k
        self.n_cluster = n_cluster

    def noise_remover(self):
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.train)
        for a_min in range(len(self.minority)):
            n_array = neighbors.kneighbors(self.minority[a_min], return_distance=False)[0]
            count = 0
            for bor in n_array:
                if self.train[bor] in self.majority:
                    count += 1
            if count == len(n_array):
                self.minority.remove(self.minority[a_min])
        for a_maj in range(len(self.majority)):
            n_array = neighbors.kneighbors(self.majority[a_maj], return_distance=False)[0]
            count = 0
            for bor in n_array:
                if self.train[bor] in self.minority:
                    count += 1
            if count == len(n_array):
                self.majority.remove(self.minority[a_maj])

    def Mod_AggCluster(self):
        self.noise_remover()
        minority_kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(self.minority)
        minority_n_labels = minority_kmeans.labels_
        minority_n_centers = minority_kmeans.cluster_centers_.tolist()
        minority_distance = []
        for i in range(len(minority_n_centers)):
            i_list = []
            for j in range(len(minority_n_centers)):
                dis = math.sqrt(np.sum(np.square(np.array(minority_n_centers[i]) - np.array([minority_n_centers[j]]))))
                i_list.append(dis)
                minority_distance.append(i_list)

        min_a, min_b, pie = 0, 0, 0
        for i in range(len(minority_distance)):                         # @此处有浪费性能
            for j in range(len(minority_distance[i])):
                if i != j and minority_distance[i][j] < pie:
                    pie = j
                    min_a, min_b = i, j
        majority_kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(self.majority)
        majority_n_centers = majority_kmeans.cluster_centers_.tolist()
        set_a = []
        for i in range(len(majority_n_centers)):
            i_to_a = math.sqrt(np.sum(np.square(np.array(majority_n_centers[i]) - np.array([minority_n_centers[min_a]]))))
            i_to_b = math.sqrt(np.sum(np.square(np.array(majority_n_centers[i]) - np.array([minority_n_centers[min_b]]))))
            distance = i_to_a + i_to_b
            if distance < pie:
                set_a.append(distance)
        if len(set_a) != 0:
            minority_distance[min_a][min_b] = 100000000000000
