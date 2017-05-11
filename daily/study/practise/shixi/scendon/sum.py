# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/24 19:04
# @Site    : 
# @File    : sum.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

def reverseAdd(n, m):
    n2 = []
    m2 = []
    if n > 70000 or n < 1 or m > 70000 or m < 1:
        return -1
    else:
        while n:
            t = n % 10
            n /= 10
            n2.append(t)
        n2.reverse()
        while m:
            p = m % 10
            m /= 10
            m2.append(p)
        m2.reverse()
        p, q = 0, 0
        for index in range(len(n2)):
            p += n2[index] * (10**index)
        for index2 in range(len(m2)):
            q += m2[index2] * (10 ** index2)
        sum2 = p+q
        return sum2

if __name__ == '__main__':
    str1 = sys.stdin.readline().split(',')
    num1 = int(str1[0])
    num2 = int(str1[1])
    g = reverseAdd(num1, num2)
    print g