import pickle
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
import random


with open('../nspst', 'rb') as f:
    nspst = pickle.load(f)
    nspst = [[item[0], item[1:5]] for item in nspst]
    nspst = dict(nspst)
    # print(nspst)

trainlabels = []
testlabels = []
trainset = []
testset = []


trainlabels, testlabels, trainset, testset = np.array(trainlabels), np.array(testlabels), np.array(trainset), np.array(testset)
print(trainlabels.shape, testlabels.shape, trainset.shape, testset.shape)

trainset, testset = preprocessing.scale(trainset), preprocessing.scale(testset)  # 数据标准化

clf = SVC()
clf.fit(trainset[:30200], trainlabels[:30200])
train_output = clf.predict(trainset)
test_output = clf.predict(testset)

def get_accuracy(output, labels):
    right = 0
    for i in range(len(testlabels)):
        if output[i] == testlabels[i]:
            right += 1
    accuracy = right / len(testlabels)
    print(accuracy)

get_accuracy(train_output, trainlabels)
get_accuracy(test_output, testlabels)
