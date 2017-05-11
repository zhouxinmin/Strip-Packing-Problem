# coding:utf-8
import numpy
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet

net = buildNetwork(24, 3, 1,  hiddenclass=TanhLayer, outclass=SoftmaxLayer)
ds = SupervisedDataSet(24, 1)
trainData = numpy.loadtxt(open("H:/Datamining/German Credit Dataset/train.csv", "rb"),	delimiter=",", skiprows=0)
for i in trainData:
    ds.addSample(i[:24], i[24])

trainer = BackpropTrainer(net, ds)

out = SupervisedDataSet(24, 1)
test = numpy.loadtxt(open("H:/Datamining/German Credit Dataset/train.csv", "rb"),	delimiter=",", skiprows=0)
for j in test:
    out.addSample(j[:24], 1)

n = FeedForwardNetwork()

outPre = net.activate(out)
for t in outPre:
    print t