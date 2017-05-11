# coding:utf-8

import csv
import numpy

fileOfTxt = open('H:/Datamining/UCI/ionosphere/ionosphere.data.txt', 'r')
csvfile = file('H:/Datamining/UCI/ionosphere/ionosphereData.csv', 'wb')
writer = csv.writer(csvfile)
for line in fileOfTxt:
    if line:
        data = line.replace("\n", "").split(",")
        print data
        writer.writerow(data)
    else:
        break
fileOfTxt.close()
csvfile.close()