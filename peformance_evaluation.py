import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""These functions take in the large restult dictionaries from the main.py apply_algorithms functions, converts the 
dictionary to a data frame and then plots all the performance metrics for each model and representation type
(resnet50, pge, inceptionv3, vggg16)
"""
# plots the scores for each model and preprocessing method against each other on a bar chart
def print_plot(df, key, representation):
    models = ['K-Means', 'GMM', 'Heirarchical Clustering', 'Louvain']

    X_axis = np.arange(len(models))

    colors = ['r', 'b', 'g', 'y']
    color_ind = 0
    width = 0.2
    pge_score = []
    res_score = []
    inc_score = []
    vgg_score = []

    # compare models against eachother for representaiton
    for col in df:
        for row in df[col]:
            if col == 'pge':
                pge_score.append(row[key])
            elif col == 'resnet50':
                res_score.append(row[key])
            elif col == 'inceptionv3':
                inc_score.append(row[key])
            elif col == 'vgg16':
                vgg_score.append(row[key])

    plt.bar(X_axis, pge_score, color='r', width=width, label = 'pge')
    plt.bar(X_axis+0.2, res_score, color='b', width=width, label = 'resnet50')
    plt.bar(X_axis+0.4, inc_score, color='g', width=width, label = 'inceptionv3')
    plt.bar(X_axis+0.6, vgg_score, color='y', width=width, label = 'vgg16')

    plt.title(key+" for "+representation)
    plt.xticks(X_axis+0.3, models)
    plt.legend()
    plt.show()


def plots(df, rep):
    keys = ['v_score', 'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    for key in keys:
        print_plot(df, key, rep)


def make_plots(result, representation):
    df = pd.DataFrame.from_dict(result)
    plots(df, representation)

