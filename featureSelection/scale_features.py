import sys
sys.path.append("../tools/")
import loadEnron
import numpy as np
import pickle

data = loadEnron.load()

def modify_profile(data, func):
    for key in data:
        data[key] = func(data[key])
    return data

# In: features list; Out: features' extent dict.
def get_extent(data, features):
    matrix = []
    for person in data:
        matrix.append([])
        for ft in features:
            matrix[-1].append(data[person][ft])
    matrix = np.array(matrix)
    min_a, max_a = np.amin(matrix, axis=0), np.amax(matrix, axis=0)
    min_max = np.swapaxes(np.vstack((min_a, max_a)), 0, 1)
    result = {}
    for i, e in enumerate(features):
        result[e] = min_max[i]
    return result

scale_vars = ['total_stock_value', 'incentives', 'salary', 
                  'deferred_income', 'expenses', 'other']
ext_dict = get_extent(data, scale_vars)

def scaler(profile, feature, ext_dict=ext_dict):
    extent = ext_dict[feature]
    new_name = "scaled_" + feature
    profile[new_name] = (profile[feature] - extent[0]) * 1.0 / (extent[1] - extent[0])
    return profile

def profile_scaler(profile, scale_fts=scale_vars):
    for feature in scale_fts:
        profile = scaler(profile, feature)
    return profile
    
data = modify_profile(data, profile_scaler)
print data[data.keys()[0]]
with open("../data/data_scaled.pkl", 'wb') as f:
    pickle.dump(data, f)

