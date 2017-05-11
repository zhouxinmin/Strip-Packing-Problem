# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/24 20:37
# @Site    : 
# @File    : 22.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

str1 = sys.stdin.readline().strip()

org = [[1, 2], [3, 4], [5, 6]]
for i in str1:
    if i == 'R':
        r = org[0]
        r.reverse()
        l = org[1]
        org[0] = l
        org[1] = r
    if i == 'L':
        r = org[0]
        l = org[1]
        l.reverse()
        org[0] = l
        org[1] = r
    if i == 'F':
        r = org[0]
        l = org[1]
        l.reverse()
        org[0] = l
        org[1] = r
    if i == 'B':
        r = org[0]
        l = org[1]
        l.reverse()
        org[0] = l
        org[1] = r
    if i == 'A':
        r = org[0]
        l = org[1]
        l.reverse()
        org[0] = l
        org[1] = r
    if i == 'C':
        r = org[0]
        l = org[1]
        l.reverse()
        org[0] = l
        org[1] = r
d = ''
for j in org[0]:
    t = str(j)
    d += t
for m in org[1]:
    t = str(m)
    d += t
for n in org[2]:
    t = str(n)
    d += t
print d
