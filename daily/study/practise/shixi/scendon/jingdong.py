# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/4/7 19:46
# @Site    : 
# @File    : jingdong.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

people = int(raw_input())
arr = raw_input()
arr_list = list(arr)
qiu_fan = []
for j in range(len(arr_list)):
    if arr_list[j] != 'X' and arr_list[j] != '#':
        cap = int(arr_list[j])
        left = j - cap
        right = j + cap+1
        if left < 0:
            left = 0
        if right > len(arr_list)-1:
            right = len(arr_list)
        for t in range(left, right):
            if arr_list[t] == 'X':
                qiu_fan.append(t)
qiu_fan = set(qiu_fan)
num = len(qiu_fan)
print num