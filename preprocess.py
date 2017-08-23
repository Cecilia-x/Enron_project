#!/usr/bin/python

import numpy as np
import pickle

with open("data/data.pkl", "rb") as data_file:
    data = pickle.load(data_file)
    
### Remove non-person cases
data.pop('TOTAL', 0)
data.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Create new features

### Modify BHATNAGAR SANJAY data, which has input error:
modify_features = ['director_fees', 'expenses', 'total_payments', 'exercised_stock_options',
                   'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
modify_num = [0, 137864, 137864, 15456290, 2604490, -2604490, 15456290]
for i, f in enumerate(modify_features):
    data['BHATNAGAR SANJAY'][f] = modify_num[i]

### Modify BELFER ROBERT data, which has input error:
br_data = {'deferred_income': -102500, 'expenses':3285, 
         'director_fees':102500, 'total_payments':3285,
         'restricted_stock':44093, 'restricted_stock_deferred':44093}
for ft in data['BELFER ROBERT']:
    if ft not in ['poi', 'email_address']:
        if ft in br_data.keys():
            data['BELFER ROBERT'][ft] = br_data[ft]
        else:
            data['BELFER ROBERT'][ft] = 0
    
### Create new features.
### 1. helpers 
def modify_profile(data, func):
    for key in data:
        data[key] = func(data[key])
    return data
    
def modify_features(data, features, func):
    for person in data:
        for ft in features:
            data[person][ft] = func(data[person][ft])       
    return data

# Take feature_matrix with 3 column as argument, apply function and create new feature.
def feature_calculator(profile, feature_matrix, func):
    for relation in feature_matrix:
        f1, f2, f3 = relation[0], relation[1], relation[2]
        profile[f1] = func(profile[f2], profile[f3])
    return profile        
        
### 2. Change all the 'NaN' to 0, and add new feature
def nan_to_num(fig):
    if fig == 'NaN':
        return 0
    return fig

features = data[data.keys()[0]].keys() # apply to all features
data = modify_features(data, features, nan_to_num)

### 3. Create new features: (1) share_recp_with_poi_ratio (2) poi_in_sentmsg_ratio
###                         (3) poi_in_recvmsg_ratio 

# For zero-division 
def subtract(n1, n2):
    if n2 != 0:
        return float(n1) / n2
    return 0
    
def calculate_ratio(profile):
    relates = [['share_recp_with_poi_ratio', 'shared_receipt_with_poi', 'to_messages'],
               ['poi_in_sentmsg_ratio', 'from_this_person_to_poi', 'from_messages'],
               ['poi_in_recvmsg_ratio', 'from_poi_to_this_person', 'to_messages']]
    profile = feature_calculator(profile, relates, subtract)
    return profile
    
data = modify_profile(data, calculate_ratio)

###    Create new features: (4) incentives (5) fix_income
def plus(a, b):
    return a + b

def calculate_plus(profile):
    relates = [['incentives', 'bonus', 'long_term_incentive'],
               ['fix_income', 'salary', 'deferral_payments', 'other']]
    profile = feature_calculator(profile, relates, plus)
    return profile

data = modify_profile(data, calculate_plus)

pickle.dump(data, open('data/data_processed.pkl', 'wb'))
