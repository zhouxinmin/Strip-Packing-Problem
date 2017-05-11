#  coding:utf-8
import csv

fileOfTxt = open('H:/Datamining/UCI/heart/processed.cleveland.data.txt', 'r')
csvfile = file('H:/Datamining/UCI/heart/processed.cleveland.data.csv', 'wb')
writer = csv.writer(csvfile)
for line in fileOfTxt:
    if line:
        # data = line.replace("\n", " ").split(" ")
        data = line.replace("\n", " ").split(",")
        print data
        writer.writerow(data)
    else:
        break
fileOfTxt.close()
csvfile.close()