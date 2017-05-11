# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/13 19:40
# @Site    : 
# @File    : Find.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import operator
target = 5
array = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
# write code here
index = -1
flage = 0
dic = {}
for i in array:
    index += 1
    for m in i:
        if target == m:
            flage = 1
    i.sort()
    dic[index] = i[0]
new_dic = sorted(dic.iteritems(), key=lambda d: d[1])
new_array = []
for j in new_dic:
    new_array.append(array[j[0]])
array = new_array
if flage == 1:
    print "true"
else:
    print "flase"
