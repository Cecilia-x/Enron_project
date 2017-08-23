"""
Module for loading Enron dataset.
"""

import sys
import pickle
sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit

def load_n_transf(features_list, filename="data/data_processed.pkl"):
    with open(filename, "r") as data_file:
        data_dict = pickle.load(data_file) 
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return labels, features
    
def load(filename="data/data_processed.pkl"):
    with open(filename, "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict
