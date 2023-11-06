import numpy as np
import plotly.graph_objects as go

import data
from clustering_algorithms.gmm import apply_gmm
from clustering_algorithms.hierarchical_clustering import apply_hierarchical_clustering
from clustering_algorithms.kmeans import apply_kmeans
from clustering_algorithms.louvain import apply_louvain
from data import TissuesData
import data as data_management
import pandas as pd
import evaluation

"""
    Data Representation (representations): 'pge', 'resnet50', 'inceptionv3', 'vgg16'
    Feature types (feature_types):  'pca', 'umap'
"""


def main():
    """ 1. Load dataset """
    data = TissuesData()  # Load data in constructor

    """ 2. Get all sample combinations (Feature types *  Data Representation) """
    datasets = data.get_all_datasets()

    """ 3. Creating testing data (samples from each set, e.g k = 200) """
    """ 4. Apply algorithm """

    # representation_testing_data (test_data, test_label)
    # kmeans_assignments
    # gmm_assignments
    # hierarchical_clustering_assignments
    # louvain_assignments

    # a. PCA
    pca_assignments = apply_algorithms(data, datasets, 'pca')
    # b. UMAP
    umap_assignments = apply_algorithms(data, datasets, 'umap')

    """ 5. Evaluate """
    for representation in data_management.representations:
        pass
        # pca_assignments

    # kmeans_counts = np.unique(assignment, return_counts=True)
    # print('Number of clusters from KMeans: %d, Presentation: %s' % (np.unique(assignment).shape[0], presentation))
    # kmeans_silhouette = evaluation.find_silhouette_score(test_data, assignment)
    # kmeans_v_measure = evaluation.find_v_measure(test_label, assignment)
    # pd.DataFrame({'Metrics': ['silhouette', 'V-measure'], 'Kmeans': [kmeans_silhouette, kmeans_v_measure],
    #               'Louvain': [louvain_silhouette, louvain_v_measure]}).set_index('Metrics')

    # 5. Visualize Results








def apply_algorithms(data, datasets, feature_type):
    kmeans_assignments = {}
    gmm_assignments = {}
    hierarchical_clustering_assignments = {}
    louvain_assignments = {}
    representation_testing_data = {}

    for representation in data_management.representations:
        dataset = datasets[representation][feature_type]
        test_data, test_label = data.get_testing_data(dataset, representation)
        representation_testing_data[representation] = test_data, test_label
        # K-means
        kmeans_assignment = apply_kmeans(test_data)
        kmeans_assignments[representation] = kmeans_assignment

        # GMM
        gmm_assignment = apply_gmm(test_data)
        gmm_assignments[representation] = gmm_assignment

        # Hierarchical Clustering
        print("For Representation: ", representation)
        hc_assignment = apply_hierarchical_clustering(test_data, test_label)
        hierarchical_clustering_assignments[representation] = hc_assignment



        # Louvain
        louvain_assignment = apply_louvain(test_data)
        louvain_assignments[representation] = louvain_assignment

        # location of this return call is exiting the for loop after a single iteration
        # return representation_testing_data, kmeans_assignments, gmm_assignments, hierarchical_clustering_assignments, louvain_assignments,


def plot_data(test_data, test_label, labels):
    traces = []
    for name in np.unique(labels):
        trace = go.Scatter3d(
            x=test_data[test_label == name, 0],
            y=test_data[test_label == name, 1],
            z=test_data[test_label == name, 2],
            mode='markers',
            name=name,
            marker=go.scatter3d.Marker(
                size=4,
                opacity=0.8
            )

        )
        traces.append(trace)

    data_plotted = go.Data(traces)
    layout = go.Layout(
        showlegend=True,
        scene=go.Scene(
            xaxis=go.layout.scene.XAxis(title='PC1'),
            yaxis=go.layout.scene.YAxis(title='PC2'),
            zaxis=go.layout.scene.ZAxis(title='PC3')
        )
    )
    fig = go.Figure(data=data_plotted, layout=layout)
    fig.update_layout(
        # title="First 3 pricipal components of PathologyGAN's PCA feature",
        title="First 3 pricipal components of {Representation's} {Feature} feature",
        legend_title="Legend Title",
    )

    fig.show()


if __name__ == '__main__':
    main()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# kmeans_model = KMeans(n_clusters=3, random_state=0)  # GaussianMixture(), AgglomerativeClustering(), Louvain
# kmeans_assignment = kmeans_model.fit_predict(test_data)
#
# louvain_model = Louvain(resolution=0.9, modularity='Newman', random_state=0)
# adjacency_matrix = sparse.csr_matrix(MinMaxScaler().fit_transform(-pairwise_distances(test_data)))
# louvain_assignment = louvain_model.fit_transform(adjacency_matrix)
#
# print(
#     'Number of clusters from KMeans: %d and from Louvain: %d' % (
#         np.unique(kmeans_assignment).shape[0], np.unique(louvain_assignment).shape[0]
#     )
# )
#
# kmeans_counts = np.unique(kmeans_assignment, return_counts=True)
# louvain_counts = np.unique(louvain_assignment, return_counts=True)
#
# print('Kmeans assignment counts')
# print(
#     pd.DataFrame(
#         {'Cluster Index': kmeans_counts[0], 'Number of members': kmeans_counts[1]}
#     ).set_index('Cluster Index')
# )
#
# print('Louvain assignment counts')
# print(
#     pd.DataFrame(
#         {'Cluster Index': louvain_counts[0], 'Number of members': louvain_counts[1]}
#     ).set_index('Cluster Index')
# )
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
