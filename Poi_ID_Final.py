
# coding: utf-8

# In[116]:

#!/usr/bin/python


import pandas as pd
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'deferral_payments','deferred_income', 'director_fees',
                 'exercised_stock_options','expenses', 'from_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'loan_advances', 'long_term_incentive', 'other',
                 'restricted_stock', 'restricted_stock_deferred',
                 'salary', 'shared_receipt_with_poi', 'to_messages',
                 'total_payments', 'total_stock_value'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)






# In[117]:

data_dict.pop('TOTAL',0)


# In[118]:

data_dict.pop('LOCKHART EUGENE E',0)


# In[119]:

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


# In[120]:

my_dataset = pd.DataFrame(data_dict)


# There were 146 data points in this data set and now there are 143 data points.

# In[121]:

my_dataset.info()


# In[122]:

my_dataset_explorer = my_dataset.transpose()


# In[123]:

my_dataset_explorer.info()


# There are 21 features including POI.

# In[124]:

my_dataset_explorer.groupby('poi')[['poi']].count()


# There are only 18 POI's out of the 143 records

# In[125]:

#remove text feature
email_df = my_dataset_explorer['email_address'].astype(str)
my_dataset_explorer = my_dataset_explorer.drop('email_address', axis = 1)


# In[126]:

# Seperate the POI data to use for classifiers
poi_df = my_dataset_explorer['poi'].astype(int)
poi_df = pd.DataFrame(poi_df)
poi_df.head()


# In[127]:

my_dataset_explorer = my_dataset_explorer.drop('poi', axis = 1)


# In[128]:

my_dataset_explorer = my_dataset_explorer.replace('NaN',0.000000001)


# In[129]:

my_dataset_explorer.apply(pd.to_numeric)


# In[130]:

my_dataset_explorer.corrwith(poi_df['poi'])


# The following have very little correlation with POI, so I will remove these features in addition to email_address.
# restricted_stock_deferred   -0.021548
# from_messages               -0.034671
# deferral_payments           -0.039880
# 
# I now will have 17 features to explore.
# 
# I will run PCA on the six features that have the highest correlation:
# exercised_stock_options      0.386853
# total_stock_value            0.382623
# bonus                        0.358486
# salary                       0.338851
# long_term_incentive          0.256405
# deferred_income             -0.274150

# In[131]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income'] 


# In[132]:

my_dataset = my_dataset.replace('NaN',0.000000001)


# In[133]:

my_dataset.loc['cash_payments'] = (my_dataset.loc['bonus']  
                                    + my_dataset.loc['salary']
                                    + my_dataset.loc['exercised_stock_options']
                                    + my_dataset.loc['director_fees'])


# In[134]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income', 'cash_payments'] 


# In[135]:

data = featureFormat(my_dataset, features_list, sort_keys=True)


# In[136]:

labels, features = targetFeatureSplit(data)


# In[137]:

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[138]:

from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

scaler = MinMaxScaler()
scaled_features_train = scaler.fit_transform(features_train)
scaled_features_test = scaler.fit_transform(features_test)


# In[139]:

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score


# In[140]:

from sklearn.decomposition import PCA


# In[141]:

pca = PCA(n_components = 3)
clf = GaussianNB()


# In[142]:

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
sss = StratifiedShuffleSplit(100, test_size=0.5, random_state=42)


# In[143]:

pipeline = Pipeline([("pca", pca), ("classifier", clf)])


# In[144]:

param_grid = dict(pca__n_components=range(1,8))
                 


# In[145]:

grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv = sss)


# In[146]:

grid_fit = grid_search.fit(scaled_features_train, labels_train)


# In[147]:

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

first_clf = grid_fit.best_estimator_


# In[148]:

print first_clf


# In[149]:

from sklearn.metrics import precision_recall_fscore_support
pred_train_gb = first_clf.predict(scaled_features_train)
gb_train_acc = accuracy_score(pred_train_gb, labels_train)
gb_train_all = precision_recall_fscore_support(pred_train_gb, labels_train)
pred_test_gb = first_clf.predict(scaled_features_test)
gb_test_acc = accuracy_score(pred_test_gb, labels_test)


# In[150]:

from sklearn.ensemble import RandomForestClassifier


# In[151]:

clf2 = RandomForestClassifier()


# In[152]:

pipeline2 = Pipeline([("pca", pca), ("classifier", clf2)])


# In[153]:

param_grid2 = dict(pca__n_components=range(1,8),
                 classifier__min_samples_split = (3, 50))


# In[154]:

grid_search2 = GridSearchCV(pipeline2, param_grid=param_grid2, scoring='f1', cv = sss)


# In[155]:

grid_fit2 = grid_search2.fit(scaled_features_train, labels_train)


# In[156]:

second_clf = grid_fit2.best_estimator_


# In[157]:

print second_clf


# In[158]:

pred_train_dt = second_clf.predict(scaled_features_train)
dt_train_acc = accuracy_score(pred_train_dt, labels_train)
dt_train_all = precision_recall_fscore_support(pred_train_dt, labels_train)
pred_test_dt = second_clf.predict(scaled_features_test)
dt_test_acc = accuracy_score(pred_test_dt, labels_test)


# In[159]:

print ("gb_train_acc: "), gb_train_acc
print ("gb_test_acc: "), gb_test_acc
print ("gb_train_precision_recall_fscore_support: "), gb_train_all
print ("dt_train_acc: "), dt_train_acc
print ("dt_test_acc: "), dt_test_acc
print ("dt_train_precision_recall_fscore_support: "), dt_train_all


# In[160]:

dump_classifier_and_data(first_clf, my_dataset, features_list)


# In[161]:

from tester import test_classifier
test_classifier(first_clf, my_dataset, features_list)


# I wonder if removing the feature that I engineered (cash_payments) would improve the scores.  Let's try it.

# In[162]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income'] 


# In[163]:

data = featureFormat(my_dataset, features_list, sort_keys=True)


# In[164]:

labels, features = targetFeatureSplit(data)


# In[165]:

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[166]:

scaler = MinMaxScaler()
scaled_features_train = scaler.fit_transform(features_train)
scaled_features_test = scaler.fit_transform(features_test)


# In[167]:

pca3 = PCA(n_components = 3)
clf3 = GaussianNB()


# In[168]:

pipeline3 = Pipeline([("pca", pca3), ("classifier", clf3)])


# In[169]:

param_grid3 = dict(pca__n_components=range(1,7))


# In[170]:

grid_search3 = GridSearchCV(pipeline3, param_grid=param_grid3, scoring='f1', cv = sss)


# In[171]:

grid_fit3 = grid_search3.fit(scaled_features_train, labels_train)


# In[172]:

third_clf = grid_fit.best_estimator_
print third_clf


# In[173]:

dump_classifier_and_data(third_clf, my_dataset, features_list)


# In[174]:

from tester import test_classifier
test_classifier(third_clf, my_dataset, features_list)


# Just as I suspected, removing my engineered feature improved the scores!
