
# coding: utf-8

# In[24]:

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






# In[25]:

data_dict.pop('TOTAL',0)


# In[26]:

data_dict.pop('LOCKHART EUGENE E',0)


# In[27]:

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


# In[28]:

my_dataset = pd.DataFrame(data_dict)


# There were 146 data points in this data set and now there are 143 data points.

# In[29]:

my_dataset.info()


# In[30]:

my_dataset_explorer = my_dataset.transpose()


# In[31]:

my_dataset_explorer.info()


# There are 21 features including POI.

# In[32]:

my_dataset_explorer.groupby('poi')[['poi']].count()


# There are only 18 POI's out of the 143 records

# In[33]:

#remove text feature
email_df = my_dataset_explorer['email_address'].astype(str)
my_dataset_explorer = my_dataset_explorer.drop('email_address', axis = 1)


# In[34]:

# Seperate the POI data to use for classifiers
poi_df = my_dataset_explorer['poi'].astype(int)
poi_df = pd.DataFrame(poi_df)
poi_df.head()


# In[35]:

my_dataset_explorer = my_dataset_explorer.drop('poi', axis = 1)


# In[36]:

my_dataset_explorer = my_dataset_explorer.replace('NaN',0.000000001)


# In[37]:

my_dataset_explorer.apply(pd.to_numeric)


# In[38]:

my_dataset_explorer.corrwith(poi_df['poi'])


# In[39]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income'] 


# In[40]:

my_dataset = my_dataset.replace('NaN',0.000000001)


# In[41]:

my_dataset.loc['cash_payments'] = (my_dataset.loc['bonus']  
                                    + my_dataset.loc['salary']
                                    + my_dataset.loc['exercised_stock_options']
                                    + my_dataset.loc['director_fees'])


# In[42]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income', 'cash_payments'] 


# In[43]:

data = featureFormat(my_dataset, features_list, sort_keys=True)


# In[44]:

labels, features = targetFeatureSplit(data)


# In[45]:

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[46]:

from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

scaler = MinMaxScaler()
scaled_features_train = scaler.fit_transform(features_train)
scaled_features_test = scaler.fit_transform(features_test)


# In[47]:

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score


# In[48]:

from sklearn.decomposition import PCA


# In[49]:

pca = PCA(n_components = 3)
clf = GaussianNB()


# In[50]:

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
sss = StratifiedShuffleSplit(100, test_size=0.5, random_state=42)


# In[51]:

pipeline = Pipeline([("pca", pca), ("classifier", clf)])


# In[52]:

param_grid = dict(pca__n_components=range(1,8))
                 


# In[53]:

grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv = sss)


# In[54]:

grid_fit = grid_search.fit(scaled_features_train, labels_train)


# In[55]:

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

first_clf = grid_fit.best_estimator_


# In[56]:

print first_clf


# In[57]:

from sklearn.metrics import precision_recall_fscore_support
pred_train_gb = first_clf.predict(scaled_features_train)
gb_train_acc = accuracy_score(pred_train_gb, labels_train)
gb_train_all = precision_recall_fscore_support(pred_train_gb, labels_train)
pred_test_gb = first_clf.predict(scaled_features_test)
gb_test_acc = accuracy_score(pred_test_gb, labels_test)


# In[58]:

from sklearn.ensemble import RandomForestClassifier


# In[59]:

clf2 = RandomForestClassifier()


# In[60]:

pipeline2 = Pipeline([("pca", pca), ("classifier", clf2)])


# In[61]:

param_grid2 = dict(pca__n_components=range(1,8),
                 classifier__min_samples_split = (3, 50))


# In[62]:

grid_search2 = GridSearchCV(pipeline2, param_grid=param_grid2, scoring='f1', cv = sss)


# In[63]:

grid_fit2 = grid_search2.fit(scaled_features_train, labels_train)


# In[64]:

second_clf = grid_fit2.best_estimator_


# In[65]:

print second_clf


# In[66]:

pred_train_dt = second_clf.predict(scaled_features_train)
dt_train_acc = accuracy_score(pred_train_dt, labels_train)
dt_train_all = precision_recall_fscore_support(pred_train_dt, labels_train)
pred_test_dt = second_clf.predict(scaled_features_test)
dt_test_acc = accuracy_score(pred_test_dt, labels_test)


# In[67]:

print ("gb_train_acc: "), gb_train_acc
print ("gb_test_acc: "), gb_test_acc
print ("gb_train_precision_recall_fscore_support: "), gb_train_all
print ("dt_train_acc: "), dt_train_acc
print ("dt_test_acc: "), dt_test_acc
print ("dt_train_precision_recall_fscore_support: "), dt_train_all


# In[68]:

dump_classifier_and_data(first_clf, my_dataset, features_list)


# In[69]:

from tester import test_classifier
test_classifier(first_clf, my_dataset, features_list)


# I wonder if removing the feature that I engineered (cash_payments) would improve the scores.  Let's try it.

# In[70]:

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'long_term_incentive', 'deferred_income'] 


# In[71]:

data = featureFormat(my_dataset, features_list, sort_keys=True)


# In[72]:

labels, features = targetFeatureSplit(data)


# In[73]:

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[74]:

scaler = MinMaxScaler()
scaled_features_train = scaler.fit_transform(features_train)
scaled_features_test = scaler.fit_transform(features_test)


# In[75]:

pca3 = PCA(n_components = 3)
clf3 = GaussianNB()


# In[76]:

pipeline3 = Pipeline([("pca", pca3), ("classifier", clf3)])


# In[77]:

param_grid3 = dict(pca__n_components=range(1,7))


# In[78]:

grid_search3 = GridSearchCV(pipeline3, param_grid=param_grid3, scoring='f1', cv = sss)


# In[79]:

grid_fit3 = grid_search3.fit(scaled_features_train, labels_train)


# In[80]:

third_clf = grid_fit.best_estimator_
print third_clf


# In[81]:

dump_classifier_and_data(third_clf, my_dataset, features_list)


# In[82]:

from tester import test_classifier
test_classifier(third_clf, my_dataset, features_list)


# Just as I suspected, removing my engineered feature improved the scores!
