# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/3/23 15:06
# @Site    : 
# @File    : majiang.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import sys
import copy


def huPai(a_list):
    num_ma = 0
    for value2 in a_list:
        num_ma += value2
    if num_ma != 0:
        # if num_ma % 3 == 2:
        #     for v in a_list:
        #         if v >= 2:
        #             v -= 2
        #             flag2 = huPai(a_list)
        #             if flag2 == 1:
        #                 return 1
        #             v += 2
        for k in range(len(a_list)):
            if a_list[k] >= 3:
                a_list[k] -= 3
                flag3 = huPai(a_list)
                if flag3 == 1:
                    return 1
                a_list[k] += 3
        for j in range(7):
            if a_list[j] > 0 and a_list[j+1] > 0 and a_list[j+2] > 0:
                a_list[j] -= 1
                a_list[j+1] -= 1
                a_list[j+2] -= 1
                flag4 = huPai(a_list)
                if flag4 == 1:
                    return 1
                a_list[j] += 1
                a_list[j+1] += 1
                a_list[j+2] += 1
    else:
        return 1


num = sys.stdin.readline().strip()
num_list = list(num)
indict = 0
three = len(num_list)-2
if three % 3 == 0:
    mj_list = [0]*9
    print num_list
    for i in num_list:
        t = int(i)
        mj_list[t-1] += 1
    print mj_list
    for index in range(len(mj_list)):
        if mj_list[index] >= 2:
            mj_list[index] -= 2
            new_dic = copy.deepcopy(mj_list)
            print new_dic
            indict = huPai(new_dic)
            if indict == 1:
                print "yes\n"
            else:
                mj_list[index] += 2
                print mj_list
    if indict == 1:
        print "yes\n"
    else:
        print "no\n"
else:
    print "no\n"
