# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/24 15:53
# @Site    : 
# @File    : lengstr.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

count = 0
str1 = sys.stdin.readline().split()
list1 = list(str1)
a = int(list1[0])
b = int(list1[1])
c = int(list1[2])
t = []
for i in range(a, b+1):
    t.append(i)
for j in t:
    if j % c == 0:
        count += 1
print count

