#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
#########################################################
from sklearn import svm
clf = svm.SVC(kernel = 'rbf', gamma = 'auto' ,C = 10000)

t0 = time()
clf.fit(features_train , labels_train )
print "fitting time :",round((time()-t0),3),"s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time :",round((time()-t0),3),"s"


t0 = time()
accuracy = clf.score(features_test , labels_test)
print "accuracy time :",round((time()-t0),3),"s"
print "Accuracy : " , accuracy


# Chris Email prediction
count = 0
for i in range(len(pred)):
    if pred[i] == 1 :
        count += 1
print"Chris favored =",count

# print "10",pred[10]
# print "26",pred[26]
# print "50",pred[50]




#########################################################
# inference --
# at c = 10 , accu = 0.6160409556313993
# at c = 100 , accu = 0.6160409556313993
# at c = 1000 , accu =0.8213879408418657
# at c = 10000 , accu = 0.8924914675767918
