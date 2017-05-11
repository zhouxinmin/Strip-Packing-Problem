# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/4/7 20:13
# @Site    : 
# @File    : jingdong2.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm
import sys

num_car = int(raw_input())
cars = []
for times in range(num_car):
    car = sys.stdin.readline().split()
    new_car = [int(x) for x in car]
    new_car[1] += new_car[0]
    cars.append(new_car)
