# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/24 16:24
# @Site    : 
# @File    : tranangle.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

n = int(sys.stdin.readline().strip())
str1 = sys.stdin.readline().split()
num_list = []
list1 = list(str1)
one = []
for i in list1:
    num_list.append(int(i))
for j in range(len(num_list)):
    for t in range(j+1, len(num_list)):
        one_side = num_list[j]
        two_side = num_list[t]
        num_list.remove(one_side)
        num_list.remove(two_side)
        a = num_list[0]
        b = num_list[1]
        c = num_list[2]
        if a + b > c and a + c > b and b+c > a:
            m = [a, b, c]
            mo = sorted(m)
            if mo not in one:
                one.append(mo)
        num_list.append(one_side)
        num_list.append(two_side)
num = len(one)
print num
