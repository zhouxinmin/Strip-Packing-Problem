# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/24 18:57
# @Site    : 
# @File    : 24.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

def trans(num, x):
    total = 0
    if len(x) == 1:
        if abs(num - x[0]) < 0.001:
            return 1
        else:
            return 0
    else:
        for i in range(len(x)):
            a = x[i]
            b = x[:]
            b.pop(i)
            print a, b, num
            total += trans(num - a, b) + trans(num + a, b) + trans(num * a, b) + trans(num / a, b)
        return total


while True:
    try:
        nums = raw_input().strip().split()
        num = [float(i) for i in nums]
        total = trans(24.0, num)
        if total == 0:
            print 'false'
        else:
            print 'true'

    except:
        break
