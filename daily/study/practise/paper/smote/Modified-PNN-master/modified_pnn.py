# coding:utf-8
# ! /Library/Frameworks/Python.framework/Versions/7.1/Resources/Python.app/Contents/MacOS/Python

from __future__ import division
import math
import csv
from copy import deepcopy
from sklearn.metrics import roc_curve, auc

# import data (last column is diagnosis values (0 or 1))
csv1 = list(csv.reader(open('H:/Datamining/German Credit Dataset/train.csv', 'rU'), delimiter=','))
csv2 = list(csv.reader(open('H:/Datamining/German Credit Dataset/test.csv', 'rU'), delimiter=','))
total = [map(float, x) for x in csv1]           # train data
test = [map(float, x) for x in csv2]            # test data

# test1 has test data without labels
test1 = []
for i in range(len(test)):
    test1.append(test[i][0:len(test[i])-1])

# scale test
scaledtest1 = deepcopy(test1)
mintest1 = [min([test1[x][y] for x in range(len(test1))]) for y in range(len(test1[1]))]
maxtest1 = [max([test1[x][y] for x in range(len(test1))]) for y in range(len(test1[1]))]
for x in range(len(test1)):
    for y in range(len(test1[1])):
        scaledtest1[x][y] = (test1[x][y] - mintest1[y])/(maxtest1[y] - mintest1[y])

    
case0total = []
case1total = []
# Determine the case type
for i in range(len(total)):
    if total[i][len(total[i])-1] == 1:
        case1total.append(total[i][0:-1])
    else:
        case0total.append(total[i][0:-1])


# scale data using (x-min)/(min-max) for each feature in case1total and case0total seperately
scaledcase1total = deepcopy(case1total)
scaledcase0total = deepcopy(case0total)
mincase1total = [min([case1total[x][y] for x in range(len(case1total))]) for y in range(len(case1total[1]))]
maxcase1total = [max([case1total[x][y] for x in range(len(case1total))]) for y in range(len(case1total[1]))]
mincase0total = [min([case0total[x][y] for x in range(len(case0total))]) for y in range(len(case0total[1]))]
maxcase0total = [max([case0total[x][y] for x in range(len(case0total))]) for y in range(len(case0total[1]))]

for x in range(len(case1total)):
    for y in range(len(case1total[1])):
        scaledcase1total[x][y] = (case1total[x][y] - mincase1total[y])/(maxcase1total[y] - mincase1total[y])

for x in range(len(case0total)):
    for y in range(len(case0total[1])):
        scaledcase0total[x][y] = (case0total[x][y] - mincase0total[y])/(maxcase0total[y] - mincase0total[y])

# define the probabilities and losses
# case0prob = 1
# case1prob = 1
# losscase0 = 1
# losscase1 = 1

sigmas = [1, 1]


# pattern layer pdf function
def pdf(xi, x, sigma):
    return sum([math.exp(-(sum([(xi[y][z]-x[z])**2 for z in range(len(xi[y]))])/(sigma**2))) for y in range(len(xi))])/len(xi)

# need to loop(list comprehension?) around test and sigma full lists (only uses 1 sublist at a time)


def pdf2(train, test, sigma):
    return sum([math.exp(-sum([((train[a][x]-test[x])**2)/((sigma[a][x])**2) for x in range(len(test))])) for a in range(len(train))])

# finding average of each feature
avgscaledcase1total = [] 
avgscaledcase0total = [] 
for y in range(len(scaledcase0total[1])):
    avgscaledcase1total.append(sum([scaledcase0total[x][y] for x in range(len(scaledcase0total))])/len(scaledcase0total))
for y in range(len(scaledcase1total[1])):
    avgscaledcase0total.append(sum([scaledcase1total[x][y] for x in range(len(scaledcase1total))])/len(scaledcase1total))

# print avgscaledcase0total
# print avgscaledcase1total

# finding differences between average and values
diffscaledcase0total = deepcopy(scaledcase0total)
diffscaledcase1total = deepcopy(scaledcase1total)
for x in range(len(scaledcase0total)):
    for y in range(len(scaledcase0total[1])):
        diffscaledcase0total[x][y] = avgscaledcase0total[y] - scaledcase0total[x][y]
for x in range(len(scaledcase1total)):
    for y in range(len(scaledcase1total[1])):
        diffscaledcase1total[x][y] = avgscaledcase1total[y] - scaledcase1total[x][y]

# print diffscaledcase0total
# print diffscaledcase1total


def sigmascale(mean, value, sigma):
    return math.exp(-((mean - value)**2)/sigma**2)

# compute the dis to each feature
sigmascalingvarcase0 = deepcopy(diffscaledcase0total)
for x in range(len(diffscaledcase0total)):
    for y in range(len(diffscaledcase0total[1])):
        sigmascalingvarcase0[x][y] = sigmascale(avgscaledcase0total[y], diffscaledcase0total[x][y], .5)

sigmascalingvarcase1 = deepcopy(diffscaledcase1total)
for x in range(len(diffscaledcase1total)):
    for y in range(len(diffscaledcase1total[1])):
        sigmascalingvarcase1[x][y] = sigmascale(avgscaledcase1total[y], diffscaledcase1total[x][y], .5)

# initialize lists    
roclist=[]
caseclass = []
initialsigma = 1
# case0prob = 1
# case1prob = 1
case0prob = float(len(case0total)/(len(case1total)+len(case0total)))
case1prob = float(len(case1total)/(len(case1total)+len(case0total)))
losscase0 = 1
losscase1 = 1

# create scaled sigmalists with +0.1 to avoid div by 0 errors
finalsigmascalingvarcase0 = [[(sigmascalingvarcase0[x][y]*initialsigma)+0.1 for y in range(len(sigmascalingvarcase0[x]))] for x in range(len(sigmascalingvarcase0))]
finalsigmascalingvarcase1 = [[(sigmascalingvarcase1[x][y]*initialsigma)+0.1 for y in range(len(sigmascalingvarcase1[x]))] for x in range(len(sigmascalingvarcase1))]

# print scaledtest1               # 归一化后的测试数据
# print scaledcase0total          # 归一化后的类别为0的数据
# print scaledcase1total          # 归一化后的类别为1的数据


case0class = [pdf2(scaledcase0total, scaledtest1[c], finalsigmascalingvarcase0) for c in range(len(test1))]
case1class = [pdf2(scaledcase1total, scaledtest1[c], finalsigmascalingvarcase1) for c in range(len(test1))]
case0class1 = [case0class[x]*case0prob*losscase0 for x in range(len(case0class))]
case1class1 = [case1class[x]*case1prob*losscase1 for x in range(len(case1class))]
caseclass.append(case0class)
caseclass.append(case1class)
results = [0 if case0class1[x] > case1class1[x] else 1 for x in range(len(case0class1))]
fpr, tpr, thresholds = roc_curve([test[d][-1] for d in range(len(test))], results)
roc_auc = auc(fpr, tpr)
# print [test[d][len(test[1])-1] for d in range(len(test))]
# print results
print 'case0class1', case0class1
print 'case1class1', case1class1
print 'roc_auc', roc_auc
 

print 'results', results
print 'test', [int(test[x][-1]) for x in range(len(test))]
