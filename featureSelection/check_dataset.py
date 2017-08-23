"""
This module inspects 0 value of the dataset.
"""

import sys
sys.path.append("../tools/")
import loadEnron

DATA = loadEnron.load()

def null_cases(data):
    result = []
    for key in data:
        counter = 0
        for ft in data[key]:
            if data[key][ft] == 0:
                counter += 1
        result.append([key, counter])
    return result
        
def null_feature(data):
    features = data[data.keys()[0]].keys()
    result = []
    for ft in features:
        counter = 0
        for key in data:
            if data[key][ft] == 0:
                counter += 1
        result.append([ft, counter])
    return result
        
def report(data=DATA):
    result1 = null_cases(data)
    result2 = null_feature(data)
    ft_length = len(data[data.keys()[0]])
    person_num = len(data)
    print "Total Features Number: %d" % ft_length
    for line in result1:
        if line[1] == ft_length:
            print "%s, %s" % (line[0], str(line[1]))
    print "Total Person Number: %d" % person_num
    for ft in result2:
        if ft[1] > person_num * .8:
            print "%s, %d" % (ft[0], ft[1])
            
report()
"""
Printed out result:
  Total Features Number: 26
  LOCKHART EUGENE E, 26
  Total Person Number: 144
  poi, 126
  restricted_stock_deferred, 127
  loan_advances, 141
  director_fees, 129
"""
