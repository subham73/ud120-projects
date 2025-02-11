#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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


##############################
### your code goes here ###
length = len(features_train[0])
print "length : " , length
###########################
#########################################################

from sklearn import tree 
from sklearn.metrics import accuracy_score



clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf = clf.fit( features_train , labels_train )
print "fitting time :",round((time()-t0),3),"s"

t0 = time()
pred = clf.predict( features_test )
print "prediction time :",round((time()-t0),3),"s"

t0 = time()
accuracy = accuracy_score( pred , labels_test )
print "accuracy time :",round((time()-t0),3),"s"

print "accuracy : " , accuracy 


#when selectPercentile is 1 
#training time : 3.829 s
#no . of fearures : 379
# ----------------------------
#when selectPercentile is 10
#training time : 125.378 s 
#no . of fearures : 3785