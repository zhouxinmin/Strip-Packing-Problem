# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/21 9:14
# @Site    : 
# @File    : randomSmote.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class RandomSmote:
    def __init__(self, samples, n=10, k=5):
        self.n_samples, self.n_attribute = np.shape(samples)
        self.N = n
        self.k = k
        self.samples = samples
        self.new_index = 0
        self.synthetic = []

    def over_sampling(self):
        if self.N < 100:
            old_n_samples = self.n_samples
            print "old_n_samples", old_n_samples
            self.n_samples = int(float(self.N)/100*old_n_samples)
            print "n_samples", self.n_samples
            keep = np.random.permutation(old_n_samples)[:self.n_samples]
            print "keep", keep
            new_samples = self.samples[keep]
            print "new_samples", new_samples
            self.samples = new_samples
            print "self.samples", self.samples
            self.N = 100

        n = int(self.N/100)     # 每个少数类样本应该合成的新样本个数
        self.synthetic = np.zeros((self.n_samples*n, self.n_attribute))
        # print "self.synthetic", self.synthetic

        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        # print "neighbors", neighbors
        for i in range(len(self.samples)):
            nn_array = neighbors.kneighbors(self.samples[i], return_distance=False)[0]
            # 存储k个近邻的下标
            self.__populate(n, i, nn_array)
        return self.synthetic

    # 从k个邻居中随机选取N次，生成N个合成的样本
    def __populate(self, n, sub_index, nn_array):
        for i in range(n):
            result_list = random.sample(range(0, self.k), 2)
            a1 = result_list[0]
            a2 = result_list[1]
            dif = np.array(self.samples[nn_array[a1]])-np.array(self.samples[nn_array[a2]])    # 包含类标
            gap = np.random.rand(1, self.n_attribute)
            tem = self.samples[nn_array[a2]] + gap.flatten()*dif
            dif2 = np.array(tem)-np.array(self.samples[sub_index])
            gap2 = np.random.rand(1, self.n_attribute)
            self.synthetic[self.new_index] = self.samples[sub_index]+gap2.flatten()*dif2
            self.new_index += 1
