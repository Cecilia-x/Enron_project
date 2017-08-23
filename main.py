#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


### Task 1: Select what features you'll use.
features_list = ['poi', 'scaled_expenses', 'share_recp_with_poi_ratio', 'poi_in_recvmsg_ratio', 'poi_in_sentmsg_ratio']

with open(../data/data_scaled.pkl", "r") as data_file:
    my_dataset = pickle.load(data_file)
    
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

### 1. decision tree:
def get_DT_clf():
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=5, class_weight='balanced')
    return clf

### 2. SVM and Grid search
def get_gridSearch_clf(kernel='rbf'):
    param_grid = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
    clf_gridsearch = GridSearchCV(SVC(kernel=kernel, class_weight='balanced'), param_grid)
    return clf_gridsearch

# Final algorithm
clf = SVC(C=50000.0, kernel='rbf', gamma=0.1, class_weight='balanced')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

### Report Functions
def print_report(labels_test, pred):
    print classification_report(labels_test, pred, target_names=['NOT POI', 'POI'])
    print confusion_matrix(labels_test, pred)

def print_ftImportance_report(importance, features_list):
    for i, e in enumerate(importance):
        print features_list[i+1], '%.3f' % e
    
def print_accuracy(prediction, labels_test):
    score = accuracy_score(prediction, labels_test)
    print 'accuracy: %.3f' % score
   
### Stratified shuffle train/test splitter (for 1 split):
def train_test_split(features, labels, test_size=0.3, random_state=0):
    labels, features = np.array(labels), np.array(features)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for list1, list2 in sss.split(features, labels):
        trainlist, testlist = list1, list2
    features_train, labels_train = features[trainlist], labels[trainlist]
    features_test, labels_test = features[testlist], labels[testlist]
    return features_train, features_test, labels_train, labels_test

### For rbf kernel SVM feature selection.
def get_recall(pred, labels_test):
    tp = 0.0
    fn = 0.0
    for i, e in enumerate(pred):
        if e == 1 and labels_test[i] == 1:
            tp += 1
        elif e == 0 and labels_test[i] == 1:
            fn += 1
    if tp+fn == 0:
        return 0
    else:
        return tp / (tp+fn)

def split_test_newlist(clf, dataset, test_list, nfold=3):
    accuracies, recalls = .0, .0
    data = featureFormat(my_dataset, test_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    labels, features = np.array(labels), np.array(features)
    sss = StratifiedShuffleSplit(n_splits=nfold, test_size=0.3, random_state=0)
    for trainlist, testlist in sss.split(features, labels):
        features_train, labels_train = features[trainlist], labels[trainlist]
        features_test, labels_test = features[testlist], labels[testlist]
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        accuracies += accuracy_score(pred, labels_test)
        recalls += get_recall(pred, labels_test)
    return accuracies/nfold, recalls/nfold  

def test_svc(dataset, features_list):
    bucket = features_list[1:]
    selected,recalls,accuracies = [], [], []
    while bucket != []:
        max_recall, best_feature, current_accuracy = 0,0,0
        for feature in bucket:
            test_list = ['poi']
            test_list.extend(selected[:])
            test_list.append(feature)
            clf = get_gridSearch_clf(kernel='rbf')
            accu, recall = split_test_newlist(clf, dataset, test_list)
            if recall >= max_recall:
                max_recall, best_feature, current_accuracy = recall, feature, accu
        selected.append(best_feature)
        bucket.remove(best_feature)
        recalls.append(max_recall)
        accuracies.append(current_accuracy)
    for i, e in enumerate(selected):
        print "Feature: %s, Recall: %.3f, Accuracy: %.3f" % (e, recalls[i], accuracies[i])

### For linear kernel SVM feature selection.
from sklearn.feature_selection import RFECV
def rfecv_test(features, labels, features_list):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    clf = SVC(kernel='linear', class_weight='balanced')
    rfecv = RFECV(estimator=clf, step=1, cv=10)
    rfecv.fit(features_train, labels_train)
    ranking = rfecv.ranking_
    print 'N of features: %d' % rfecv.n_features_
    for i, e in enumerate(ranking):
        print "Feature: %s, rank: %d" % (features_list[i+1], e)
  
### other tunning functions:       
### This function will try different split_num of leaves, from 1 to 10, and print the accuracies. 
def tune_tree_clf(clf, features, labels):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    best = 0
    best_split = 1
    for i in range(2,10):
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        score = accuracy_score(pred, labels_test)
        print "split: %s, score: %s" % (str(i), str(score))
        if score > best:
            best = score
            best_split = i
    print "best score:%.3f, split_num=%.3f" % (best, best_split) 
    return best_split
            
### Split Test functions.
def split_test(clf, features, labels):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=12)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print_accuracy(pred, labels_test)
    print_report(labels_test, pred)
    return clf
    
def split_test_dt(clf, features, labels, features_list):
    clf = split_test(clf, features, labels)
    f_importance = clf.feature_importances_
    print_ftImportance_report(f_importance, features_list)
    
### stratified shuffle split test:
def sss_test():
    from tester import test_classifier
    test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)
