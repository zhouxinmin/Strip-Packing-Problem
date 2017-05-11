# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/11 14:56
# @Site    : 
# @File    : wangyi.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm


import sys


def one(num_list):
    tem = []
    to_1 = num_list[0: len(num_list)/2]
    to_2 = num_list[len(num_list)/2:]
    for i in range(len(to_1)):
        tem.append(to_1[i])
        tem.append(to_2[i])
    return tem


def k_times(k_time, list_num):
    for i in range(k_time):
        new_num_list = one(list_num)
        list_num = new_num_list
    return list_num

n = int(sys.stdin.readline().strip())
for i in range(n):
    information = sys.stdin.readline().strip().split(" ")
    information_list = list(information)
    num = int(information_list[0])
    k = int(information_list[1])
    pie = sys.stdin.readline().strip().split(" ")
    num_list = list(pie)
    re_list = k_times(k, num_list)
    print (re_list)






