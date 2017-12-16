#!/usr/bin/python

import os
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'expenses', 'percentage_from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print "For EACH features, calculate the number of missing values :"
features = data_dict['METTS MARK'].keys()
persons = data_dict.keys()
dict_for_missing_values = {}
for feature in features:
    dict_for_missing_values[feature] = 0
for person in data_dict.keys():
    for feature in features:
        if data_dict[person][feature] == 'NaN':
            dict_for_missing_values[feature] += 1
for key in dict_for_missing_values.keys():
    dict_for_missing_values[key] *= 1.0
    dict_for_missing_values[key] /= len(data_dict.keys())
print dict_for_missing_values

print "Extract low NaN rate features"
current_valid_features = []
for key in dict_for_missing_values.keys():
    if dict_for_missing_values[key] < 0.5:
        current_valid_features.append(key)
print current_valid_features

final_features = []
for feature in current_valid_features:
    if feature != 'email_address' and feature != 'other': # Discard both "email_address" and "other" feature!
        final_features.append(feature)
print "Final features to go to next step is ", final_features

import pandas as pd
import numpy as np

data = np.transpose(featureFormat(data_dict, final_features, remove_all_zeroes=False))
data_df = pd.DataFrame(data, final_features).transpose()
data_df.index = data_dict.keys()
data_df.head()

data_df_2 = data_df.drop('TOTAL')
data_df_3 = data_df_2.drop('BHATNAGAR SANJAY')
data_df_without_outlier = data_df_3
data_df_without_outlier.head()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_df_without_outlier['percentage_from_poi_to_this_person'] = data_df_without_outlier.from_poi_to_this_person / data_df_without_outlier.to_messages
data_df_without_outlier['percentage_from_this_person_to_poi'] = data_df_without_outlier.from_this_person_to_poi / data_df_without_outlier.from_messages
data_df_without_outlier['percentage_from_this_person_to_poi'].fillna(0, inplace=True)
data_df_without_outlier['percentage_from_poi_to_this_person'].fillna(0, inplace=True)

### Extract features and labels from dataset for local testing
from sklearn import tree

X = data_df_without_outlier.drop('poi', axis = 1)
print list(X)
Y = data_df_without_outlier.poi
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print clf.feature_importances_

### Convert DataFrame to a dictionary of dictionary
def convert_df_to_dict_of_dict(df):
    '''
        dictionary keys are names of persons in dataset
        dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person
    '''
    list_of_person = df.index.values
    feature_list = list(df)
    rst = {}
    for person in list_of_person:
        cur_dict = {}
        for feature in feature_list:
            cur_dict[feature] = df.loc[person, feature]
        rst[person] = cur_dict
    return rst

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

my_dataset = convert_df_to_dict_of_dict(data_df_without_outlier)
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

param_grid = {'min_samples_split': np.arange(2, 10)}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)

clf = clf.fit(features_train,labels_train)
print "Best parameter for min_sample_split is:"
print clf.best_estimator_
pred= clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "accuracy = ", accuracy_score(labels_test, pred)
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.best_estimator_, my_dataset, features_list)
print "Running tester.py to test my classifier's precision and recall.... "
os.system("python tester.py")


