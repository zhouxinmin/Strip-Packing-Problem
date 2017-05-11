# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/6 10:38
# @Site    : http://www.sciencedirect.com/science/article/pii/S0957417415007356
# @File    : BorderlineSmote.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors


class BorderlineSmote:
    def __init__(self, minority, majority, k=5):
        """

        :param minority: 少数类
        :param majority: 多数类
        :param k:  最邻近数
        """
        self.minority = minority
        self.majority = majority
        train = np.row_stack((self.minority, self.majority))
        self.train = train.tolist()
        self.positive_num, self.positive_attribute = np.shape(self.minority)
        self.new_index = 0
        self.k = k
        self.s = random.randint(1, k)
        self.synthetic = []

    def generate_danger(self):
        danger = []
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.train)
        for positive in range(len(self.minority)):
            n_array = neighbors.kneighbors(self.minority[positive], return_distance=False)[0]
            count = 0
            for bor in n_array:
                if self.train[bor] in self.majority:
                    count += 1
            if float(len(n_array))/2 <= count < len(n_array):
                danger.append(positive)
        return danger

    def over_sampling(self):
        danger_list = self.generate_danger()
        num_danger, num_attribute = np.shape(self.minority)
        self.synthetic = np.zeros((num_danger * self.s, num_attribute))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.minority)
        for danger in danger_list:
            nn_array = neighbors.kneighbors(self.minority[danger], return_distance=False)[0]
            self.populate(danger, nn_array)
        return self.synthetic

    def populate(self, sub_index, nn_array):
        for i in range(self.s):
            nn = np.random.randint(0, self.k)
            dif = np.array(self.minority[nn_array[nn]]) - np.array(self.minority[sub_index])  # 包含类标
            gap = np.random.rand(1, self.positive_attribute)
            self.synthetic[self.new_index] = self.minority[sub_index] + gap.flatten() * dif
            self.new_index += 1
