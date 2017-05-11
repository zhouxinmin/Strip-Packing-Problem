# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/25 13:50
# @Site    : 
# @File    : first.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import sys

num = sys.stdin.readline().strip().split(' ')
num_list = list(num)
m = int(num_list[0])
k = int(num_list[1])
num2 = sys.stdin.readline().strip().split(' ')
num2_list = list(num2)
num2_list = [int(a) for a in num2_list]
for i in range(k):
    first = num2_list[0]
    for j in range(len(num2_list)-1):
        num2_list[j] += num2_list[j+1]
    num2_list[-1] += first
    for t in range(len(num2_list)):
        if num2_list[t] >= 100:
            num2_list[t] %= 100
num_list = ''
for c in num2_list:
    s = str(c)
    num_list += s
    num_list += ' '
num_list = num_list[:-1]
print num_list
