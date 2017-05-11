# coding:utf-8
import csv
import numpy
import time

bankCsv = numpy.loadtxt(open('H:/Datamining/rong360/credit/train/bank_detail_train.csv', "rb"), delimiter=",", skiprows=0)
overdueCsv = numpy.loadtxt(open('H:/Datamining/rong360/credit/train/overdue_train.csv', "rb"), delimiter=",", skiprows=0)
csvfile = file('H:/Datamining/rong360/credit/train/bank_month_train.csv', 'wb')
writer = csv.writer(csvfile)
for per in overdueCsv:
    num, income, output, salary = 0, 0, 0, 0
    for deal in bankCsv:
        if deal[0] == per[0]:
            num += 0
            if deal[2] == 0 and deal[4] == 1:
                salary += deal[3]
            elif deal[2] == 0 and deal[4] != 1:
                income += deal[3]
            elif deal[2] == 1:
                output += deal[3]
    persalary = salary / 19         # 月薪水
    peroutput = output / 19         # 月消费
    perincome = income / 19         # 月收入（不包含薪水）
    data = [per[0], num, salary, income, output, persalary, peroutput, perincome]
    writer.writerow(data)