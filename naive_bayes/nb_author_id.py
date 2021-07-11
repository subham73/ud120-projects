#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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




# #########################################################
#IMP Message
#due to upadted version of sklearn , u need to change 2 lines in email_preprocess.py words_file

#line 7 to :
# from sklearn.model_selection import train_test_split

#line 42 to:
#features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

###########################################################
# now ready to go
##################---- CODE ---############################

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# accuracy = clf.score(features_test, labels_test)
# print "Accuracy : ",accuracy

##################################################

# time to train

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#way to know the time requiring to tain
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

t0 = time()
accuracy = clf.score(features_test, labels_test)
print "acuracy calulation time:", round(time()-t0, 3), "s"

print "Accuracy : ",accuracy
