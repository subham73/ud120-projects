#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time 
from sklearn.metrics import accuracy_score
print "AdaBoost"
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(learning_rate = 0.5 ,n_estimators = 24)

t0 = time()
clf = clf.fit( features_train , labels_train )
print "Training time : " , round(time()-t0,3) , "s"

t0 = time()
pred = clf.predict( features_test )
print "Prediction time : " , round(time()-t0,3) , "s" 


t0 = time()
acc = accuracy_score( pred , labels_test )
print "Accuracy time :  " , round(time()-t0,3) , "s"
print "Accuracy :" , acc 

#---------------------------------------------------------------------
print" "
print"Knn"
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=4 )

t0 = time()
clf.knn = clf_knn.fit(features_train, labels_train)
print "Training time : " , round(time()-t0,3) , "s"

t0 = time()
pred_knn = clf_knn.predict(features_test)
print "Prediction time : " , round(time()-t0,3) , "s"
print "Accuracy kNN:", accuracy_score(pred_knn , labels_test)

#---------------------------------------------------------------------
print" "
print"Random forest"
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=15, min_samples_split=6)

t0 = time()
clf_rf.fit(features_train, labels_train)
print "Training time : " , round(time()-t0,3) , "s"

t0 = time()
clf_rf = clf_rf.predict(features_test)
print "Prediction time : " , round(time()-t0,3) , "s"
print "Accuracy random forests:", accuracy_score(labels_test, clf_rf)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass



#overal--
#best model for above : knn 
#accuracy : 94%


#--------------------------------adaboost
#performance to beat: 93.6
#-------------------
#Till Now Best : 0.928 
    #parameter (learning_rate =1 , n_estimators = 20  )
#no parameters : 0.924
#learning rate
                # learning_rate =100 : 0.336
                # learning_rate =50  : 0.664
                # learning_rate =2   : 0.92
                # learning_rate =5   : 0.764
                # learning_rate =0.8  : 0.924
                # learning_rate =0.1 : 0.916
    #result : max at :1
#-----------------------------------KNN

#phase 2

# AdaBoost
# Training time :  0.031 s
# Prediction time :  0.015 s
# Accuracy time :   0.0 s
# Accuracy : 0.928

# Knn
# Training time :  0.0 s
# Prediction time :  0.0 s
# Accuracy kNN: 0.94

# Random forest
# Training time :  0.032 s
# Prediction time :  0.0 s
# Accuracy random forests: 0.92