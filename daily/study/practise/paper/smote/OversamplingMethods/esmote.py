# coding:utf-8

from sklearn.cluster import KMeans
import csv
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import numpy

csvfile = file('H:/Datamining/UCI/Segment/ESMOTE_segmentation.csv', 'wb')
writer = csv.writer(csvfile)

pimaData = numpy.loadtxt(open("H:/Datamining/UCI/Segment/segmentation.csv", "rb"), delimiter=",", skiprows=0)

train = pimaData[:1732]    # train
test = pimaData[1732:]     # test
train1 = []
train0 = []
# Determine the case type
for i in range(len(train)):
    if train[i][len(train[i])-1] == 1:
        train1.append(train[i][0:-1])
    else:
        train0.append(train[i][0:-1])

N = len(train1)/len(train0)     # 过采样倍数N

kmeans = KMeans(n_clusters=8, random_state=0).fit(train0)
klabels = kmeans.labels_
kcenters = kmeans.cluster_centers_
# print kcenters

newSample = []
for l in range(len(klabels)):
    center = kcenters[klabels[l]]       # 第l个元素对应的簇心
    sample = train0[l]                  # 第l个元素样本点
    dist = numpy.sqrt(numpy.sum(numpy.square(center - sample)))
    prod = map(lambda (a, b): a-b, zip(sample, center))
    for i in range(N):
        # ratio = float(dist/(N+1)) * (i+1)
        ratio = float(1/(N+1)) * (i+1)
        tempsample = [x * ratio for x in prod]
        fakesample = map(lambda (a, b): a+b, zip(center, tempsample))
        newSample.append(fakesample)
print len(newSample)
newSample2 = []
for sp in newSample:
    ca = []
    for s in sp:
        # t = round(s)
        t = s
        ca.append(t)
    ca.append(0)
    newSample2.append(ca)
# for sp in newSample:
#     sp.append(0)
print len(train[0])

print len(train[0]), len(newSample2[0])
train = numpy.concatenate((train, newSample2))
print len(train)
german = numpy.concatenate((train, test))
print len(german)
for data in german:
    writer.writerow(data)


# train_tr = train[:, :24]
# train_ta = train[:, 24]
# target = germandata[:750, 24]
# # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
# train_X, test_X, train_y, test_y = train_test_split(train_tr, train_ta, test_size=0.2, random_state=0)
#
# print len(train_X)




# def semi_unsupervised_clustering(self):
# self.noise_remover()
# minority_k_means = KMeans(n_clusters=self.n_cluster, random_state=0).fit(self.minority)
# minority_n_labels = minority_k_means.labels_
# minority_n_dict = {}
# minority_n_labels_set = copy.deepcopy(minority_n_labels)
# for i in minority_n_labels_set:
#     minority_n_dict[i] = []
# for index, value in enumerate(minority_n_labels):
#     minority_n_dict[value].append(index)
# minority_n_centers = minority_k_means.cluster_centers_.tolist()
# minority_distance = []
# for i in range(len(minority_n_centers)):
#     i_list = []
#     for j in range(len(minority_n_centers)):
#         dis = math.sqrt(np.sum(np.square(np.array(minority_n_centers[i]) - np.array([minority_n_centers[j]]))))
#         i_list.append(dis)
#         minority_distance.append(i_list)
# min_a, min_b, pie = self.find_min_dis(minority_distance)
# len_n_centers = len(minority_n_centers)
# majority_k_means = KMeans(n_clusters=self.n_cluster, random_state=0).fit(self.majority)
# majority_n_centers = majority_k_means.cluster_centers_.tolist()
# while len_n_centers < self.c_the:
#     distance = 1000000000
#     for i in range(len(majority_n_centers)):
#         i_to_a = math.sqrt(np.sum(np.square(np.array(majority_n_centers[i]) - np.array([minority_n_centers[min_a]]))))
#         i_to_b = math.sqrt(np.sum(np.square(np.array(majority_n_centers[i]) - np.array([minority_n_centers[min_b]]))))
#         real_distance = i_to_a + i_to_b
#         if distance < real_distance:
#             distance = real_distance
#     if distance <= pie:
#         minority_distance[min_a][min_b] = 1000000000
#         min_a, min_b, pie = self.find_min_dis(minority_distance)
#     else: