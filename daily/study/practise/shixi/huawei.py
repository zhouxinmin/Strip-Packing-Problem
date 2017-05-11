# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/10 14:27
# @Site    : 
# @File    : huawei.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys

# word_str = sys.stdin.readline().strip()
# print (type(word_str))
# word_list = word_str.split(' ')
# print (type(word_list))
# last_word = word_list[-1]
# print (len(last_word))

# a_str = sys.stdin.readline().strip()
# a_char = sys.stdin.readline().strip()
# a_list = list(a_str)
# count = 0
# for a in a_list:
#     if ord(a_char) == ord(a):
#         count += 1
#     elif (65 <= ord(a) <= 90 or 97 <= ord(a) <= 122) and (ord(a) + 32 == ord(a_char) or ord(a) - 32 == ord(a_char)):
#         count += 1
# print (count)
# tem = []
#
while True:
    try:
        tem = []
        n = int(sys.stdin.readline())
        for i in range(n):
            num = int(sys.stdin.readline())
            if num not in tem:
                tem.append(num)
        tem.sort()
        tem_len = len(tem)
        dex = 0
        while dex < tem_len:
            print (tem[dex])
            dex += 1
    except:
        break

first = sys.stdin.readline().strip()
second = sys.stdin.readline().strip()
num_first = len(first)
num_second = len(second)
if num_first % 8 != 0:
    num = num_first % 8
    first += '0' * (8-num)
if num_second % 8 != 0:
    num2 = num_second % 8
    second += '0' * (8-num2)
print (first)
for i in range(len(first)/8):
    print (first[i*8: i*8+7])
for j in range(len(second)/8):
    print (second[j*8:j*8+7])
