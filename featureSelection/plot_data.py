"""
Module for plotting data.
"""

import matplotlib.pyplot as plt


def plotVariables(data, f1, f2, output_name, x_annotate=None, y_annotate=None, excludeNan=False):
    for key in data:
        if excludeNan:
            if x == 0 or y == 0:
                continue
                
        if data[key]['poi'] == True:
            cl = "r"
        else:
            cl = 'b'
            
        x = data[key][f1]
        y = data[key][f2]
        
        plt.scatter(x, y, color=cl)
        
        last_name = key.split(" ")[0]
        if x_annotate and x > x_annotate:
            plt.annotate(last_name, xy=(x,y))
        if y_annotate and y > y_annotate:
            plt.annotate(last_name, xy=(x,y))

    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.savefig(output_name)
    plt.clf()
