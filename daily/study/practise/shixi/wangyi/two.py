# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/25 15:12
# @Site    : 
# @File    : two.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import sys

num = sys.stdin.readline().strip()
people = int(sys.stdin.readline().strip())
num_list = list(num)
num_list.reverse()
num_sun = 0
max_sum = 0
for index in range(len(num_list)):
    if num_list[index] == 'X':
        max_sum += 9 * 10 ** index
    else:
        num_sun += int(num_list[index]) * 10 ** index
        max_sum += int(num_list[index]) * 10 ** index
count = 0
for ct in range(num_sun, max_sum):
    if ct % people == 0:
        count += 1
print count
