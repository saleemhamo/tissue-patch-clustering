import matplotlib.pyplot as plt
import numpy as np

# obtained results from the main function
pca_results ={
    'pge': {
        'K-Means': {
            'v_score': 0.6826505172352025,
            'v_score_params': [9],
            'silhouette_score': -0.035054848,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.4702720476867322,
            'v_score_params': ['diag', 'kmeans', 0, True],
            'silhouette_score': 0.30364215,
            'silhouette_score_params': ['tied', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.46616190044482375,
            'v_score_params': [9, 'euclidean', 'auto', 'ward'],
            'silhouette_score': 0.28470385,
            'silhouette_score_params': [2, 'euclidean', 'auto', 'complete']},
        'Louvain': {
            'v_score': 0.4350045123514932,
            'v_score_params': [2.600000000000001, True, 'Potts'],
            'silhouette_score': 0.05663969,
            'silhouette_score_params': [0.7999999999999999, True, 'Potts']}},
    'resnet50': {
        'K-Means': {
            'v_score': 0.7500909068577803,
            'v_score_params': [9],
            'silhouette_score': 0.007592828,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.6675953521472837,
            'v_score_params': ['spherical', 'random', 0, True],
            'silhouette_score': 0.2047506,
            'silhouette_score_params': ['spherical', 'random_from_data', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.639339896282885,
            'v_score_params': [9, 'euclidean', 'auto', 'average'],
            'silhouette_score': 0.20420735,
            'silhouette_score_params': [2, 'euclidean', 'auto', 'ward']},
        'Louvain': {
            'v_score': 0.4944893121401915,
            'v_score_params': [0.1, True, 'Potts'],
            'silhouette_score': 0.10731749,
            'silhouette_score_params': [0.1, True, 'Potts']}},
    'inceptionv3': {
        'K-Means': {
            'v_score': 0.752416535484372,
            'v_score_params': [9],
            'silhouette_score': 0.048548847,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.4248974374495107,
            'v_score_params': ['full', 'k-means++', 0, True],
            'silhouette_score': 0.37663648,
            'silhouette_score_params': ['full', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.42490356770037496,
            'v_score_params': [9, 'euclidean', 'auto', 'ward'],
            'silhouette_score': 0.36288834,
            'silhouette_score_params': [2, 'euclidean', 'auto', 'average']},
        'Louvain': {
            'v_score': 0.36728298745034776,
            'v_score_params': [9.499999999999982, True, 'Potts'],
            'silhouette_score': 0.1285057,
            'silhouette_score_params': [0.1, True, 'Potts']}},
    'vgg16': {
        'K-Means': {
            'v_score': 0.7269327379223158,
            'v_score_params': [9],
            'silhouette_score': -0.01882117,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.6339790932128929,
            'v_score_params': ['spherical', 'kmeans', 0, True],
            'silhouette_score': 0.16628581,
            'silhouette_score_params': ['spherical', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.6676550171740204,
            'v_score_params': [9, 'cosine', 'auto', 'average'],
            'silhouette_score': 0.22889076,
            'silhouette_score_params': [2, 'euclidean', 'auto', 'average']},
        'Louvain': {
            'v_score': 0.3660129055169312,
            'v_score_params': [6.999999999999991, True, 'Potts'],
            'silhouette_score': 0.040425558,
            'silhouette_score_params': [0.4, True, 'Potts']
            }
        }
    }
umap_results = {
    'pge': {
        'K-Means': {
            'v_score': 0.7572559550069933,
            'v_score_params': [9],
            'silhouette_score': 0.033522338,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.640946088882115,
            'v_score_params': ['diag', 'kmeans', 0, True],
            'silhouette_score': 0.570827,
            'silhouette_score_params': ['diag', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.6302927057081651,
            'v_score_params': [9, 'euclidean', 'auto', 'complete'],
            'silhouette_score': 0.6238832,
            'silhouette_score_params': [7, 'euclidean', 'auto', 'complete']},
        'Louvain': {
            'v_score': 0.5779432496877341,
            'v_score_params': [2.0000000000000004, True, 'Potts'],
            'silhouette_score': 0.33050743,
            'silhouette_score_params': [0.9999999999999999, True, 'Dugue']}},
    'resnet50': {
        'K-Means': {
            'v_score': 0.7216244421368441,
            'v_score_params': [9],
            'silhouette_score': 0.13158932,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.7528840520914104,
            'v_score_params': ['spherical', 'k-means++', 0, True],
            'silhouette_score': 0.5984592,
            'silhouette_score_params': ['diag', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.7533025948800598,
            'v_score_params': [6, 'euclidean', 'auto', 'ward'],
            'silhouette_score': 0.6522304,
            'silhouette_score_params': [9, 'euclidean', 'auto', 'complete']},
        'Louvain': {
            'v_score': 0.6811819800787234,
            'v_score_params': [1.0999999999999999, True, 'Dugue'],
            'silhouette_score': 0.3597288,
            'silhouette_score_params': [1.0999999999999999, True, 'Dugue']}},
    'inceptionv3': {
        'K-Means': {
            'v_score': 0.7142118131503213,
            'v_score_params': [9],
            'silhouette_score': 0.13774829,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.496598109530028,
            'v_score_params': ['diag', 'kmeans', 0, True],
            'silhouette_score': 0.5439961,
            'silhouette_score_params': ['spherical', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.501971320933409,
            'v_score_params': [9, 'l1', 'auto', 'average'],
            'silhouette_score': 0.61594373,
            'silhouette_score_params': [2, 'euclidean', 'auto', 'ward']},
        'Louvain': {
            'v_score': 0.5757867657184093,
            'v_score_params': [2.0000000000000004, True, 'Potts'],
            'silhouette_score': 0.22622566,
            'silhouette_score_params': [0.9999999999999999, True, 'Dugue']}},
    'vgg16': {
        'K-Means': {
            'v_score': 0.8605238040025479,
            'v_score_params': [9],
            'silhouette_score': 0.2909466,
            'silhouette_score_params': [9]},
        'GMM': {
            'v_score': 0.7663097905718953,
            'v_score_params': ['spherical', 'random', 0, True],
            'silhouette_score': 0.6989255,
            'silhouette_score_params': ['full', 'kmeans', 0, True]},
        'Heirarchical Clustering': {
            'v_score': 0.7371828668056379,
            'v_score_params': [8, 'euclidean', 'auto', 'ward'],
            'silhouette_score': 0.6989255,
            'silhouette_score_params': [5, 'euclidean', 'auto', 'complete']},
        'Louvain': {
            'v_score': 0.6057236202811282,
            'v_score_params': [1.4000000000000001, True, 'Dugue'],
            'silhouette_score': 0.34213462,
            'silhouette_score_params': [0.9999999999999999, True, 'Dugue']
            }
        }
    }

# plots the scores for each model and preprocessing method against each other on a bar chart
def plot_results(results, key, representation):
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
    for rep in results:
      v_scores = []
      s_scores = []
      for alg in results[rep]:
        if rep == 'pge':
          pge_score.append(results[rep][alg][key])
        elif rep == 'resnet50':
          res_score.append(results[rep][alg][key])
        elif rep == 'inceptionv3':
          inc_score.append(results[rep][alg][key])
        elif rep == 'vgg16':
          vgg_score.append(results[rep][alg][key])



    plt.bar(X_axis, pge_score, color='r', width=width, label = 'pge')
    plt.bar(X_axis+0.2, res_score, color='b', width=width, label = 'resnet50')
    plt.bar(X_axis+0.4, inc_score, color='g', width=width, label = 'inceptionv3')
    plt.bar(X_axis+0.6, vgg_score, color='y', width=width, label = 'vgg16')

    plt.title(key+" for "+representation)
    plt.xticks(X_axis+0.3, models)
    plt.legend()
    plt.show()



plot_results(pca_results, 'v_score', 'PCA')
plot_results(pca_results, 'v_score', 'UMAP')
plot_results(umap_results, 'silhouette_score', 'PCA')
plot_results(umap_results, 'silhouette_score', 'UMAP')
