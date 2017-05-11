#  coding:utf-8
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import csv
import numpy

from imblearn.over_sampling import SMOTE


csvfile = file('H:/Datamining/UCI/vehicle/SMOTE_vehicle.csv', 'wb')
writer = csv.writer(csvfile)

germandata = numpy.loadtxt(open("H:/Datamining/UCI/vehicle/vehicle.csv", "rb"),
                           delimiter=",", skiprows=0)

train = germandata[:1732]    # train
test = germandata[1732:]     # test
X = train[:, :-1]
y = train[:, -1]
# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Apply regular SMOTE
sm = SMOTE(kind='svm')
X_resampled, y_resampled = sm.fit_sample(X, y)
# X_res_vis = pca.transform(X_resampled)
print len(X[0])
print len(X_resampled)
print len(y_resampled)
resampled = []
for i in range(0, len(X_resampled)):
    t = list(X_resampled[i])
    t.append(y_resampled[i])
    resampled.append(t)
print "合成后测试集长度：", len(resampled)
german = numpy.concatenate((resampled, test))
print "合成数据集长度：", len(german)
for data in german:
    writer.writerow(data)