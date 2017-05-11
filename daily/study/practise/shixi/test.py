# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/10 19:03
# @Site    : 
# @File    : test.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys
import copy

# num = sys.stdin.readline().strip()
# num_list = list(num)
# real_num = 0
# for index, value in enumerate(num_list):
#     real_num += int(value) * pow(7, len(num_list) - index-1)
# print (real_num)
import copy

# def ke_zi(num_list):
#     if num_list[0] == num_list[1] == num_list[2]:
#         return 1
#     else:
#         return 0
#
# def shun_zi(num_list):
#     if int(num_list[0]) +2 == int(num_list[1]) +1 == int(num_list[2]):
#         return 1
#     else:
#         return 0
#
# def dui_zi(num_list):
#     len_list = len(num_list)
#     tem = []
#     for index, value in enumerate(num_list):
#         while index+1 < len_list:
#             if num_list[index] == num_list[index+1]:
#                 duizi = num_list[index: index+2]
#                 last = num_list.remove(duizi)
#                 tem.append([duizi, last])


num = sys.stdin.readline().strip()
num_list = list(num)
if len(num_list) == 2:
    if num_list[0] == num_list[1]:
        print ("yes")
    else:
        print ("no")
if len(num_list) == 5:
    tem_list = copy.deepcopy(num_list)
    if len(set(tem_list)) == 5:
        print ("no")
    else:
        b = [x for x in num_list if num_list.count(x) == 1]
        c = set(b) ^ set(tem_list)
        if len(b) == 0:
            print ("yes")
        elif len(b) == 1:
            print ("no")
        elif len(b) == 2:
            print ("no")
        elif len(b) == 3:
            b.sort()
            if int(b[0])+2 == int(b[1]) +1 == int(b[2]):
                print ("yes")
            else:
                print ("no")
