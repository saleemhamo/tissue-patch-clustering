import numpy as np
import plotly.graph_objects as go
from sknetwork.clustering import Louvain
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from clustering_algorithms.gmm import apply_gmm
from clustering_algorithms.hierarchical_clustering import apply_hierarchical_clustering
from clustering_algorithms.kmeans import apply_kmeans
from clustering_algorithms.louvain import apply_louvain
from data import TissuesData
import matplotlib.pyplot as plt
import data as data_management
from sklearn.metrics import silhouette_score, v_measure_score, davies_bouldin_score, calinski_harabasz_score


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

    # # a. PCA
    # pca_assignments = apply_algorithms(data, datasets, 'pca')
    # # b. UMAP
    # umap_assignments = apply_algorithms(data, datasets, 'umap')
    # print(pca_assignments)
    # print(umap_assignments)



    """ 5. Evaluate """
    v_score_params = [9, 'spherical', 'random', 0, True] # GMM umap vgg16
    sil_score_params = [9, 'euclidean', 'auto', 'complete'] # Heirarchical CLustering, UMAP, resnet50
    db_score_params = [2.3000000000000007, True, 'Potts'] #Louvain, UMAP, resnet50
    ch_score_params = [9, 'euclidean', 'auto', 'ward'] # Heirarchical Clustering, UMAP, vgg16

    v_dataset = datasets['vgg16']['umap']
    sil_dataset = datasets['resnet50']['umap']
    db_dataset = datasets['resnet50']['umap']
    ch_dataset = datasets['vgg16']['umap']

    # model and scores for the best v_score from the testing which is GMM with Umap and vgg16
    test_data, test_label = data.get_testing_data(v_dataset, 'vgg16')
    clustering = GaussianMixture(
                        n_components=v_score_params[0],
                        covariance_type=v_score_params[1],
                        init_params=v_score_params[2],
                        random_state=v_score_params[3],
                        warm_start=v_score_params[4],
                    )
    pred_labels = clustering.fit_predict(test_data)
    vgg_gmm_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    # model and scores for the best v_score from the testing which is Heirarchical Clusterig with Umap and resnet50
    test_data, test_label = data.get_testing_data(sil_dataset, 'resnet50')
    clustering = AgglomerativeClustering(
                        n_clusters=sil_score_params[0],
                        metric=sil_score_params[1],
                        compute_full_tree=sil_score_params[2],
                        linkage=sil_score_params[3])
    pred_labels = clustering.fit_predict(test_data)
    resnet_hc_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    test_data, test_label = data.get_testing_data(db_dataset, 'resnet50')
    clustering = Louvain(
                        resolution=db_score_params[0],
                        modularity=db_score_params[2],
                        verbose=db_score_params[1],
                        random_state=0)
    pred_labels = clustering.fit_predict(test_data)
    resnet_louv_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    test_data, test_label = data.get_testing_data(ch_dataset, 'vgg16')
    clustering = AgglomerativeClustering(
                        n_clusters=ch_score_params[0],
                        metric=ch_score_params[1],
                        compute_full_tree=ch_score_params[2],
                        linkage=ch_score_params[3])
    pred_labels = clustering.fit_predict(test_data)
    vgg_hc_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]
    plot_results(vgg_gmm_scores, resnet_hc_scores, resnet_louv_scores,vgg_hc_scores)
def plot_results(score1, score2, score3, score4):
    # function inputs are 4 lists of scoress in the order of vscore, sil score, daview bouldin, calsinski harabsz
    x_axis = ['V_score', 'Silhouette Score', 'Daview Bouldin', 'Calinski Harabasz']
    # X_axis = np.arange(len(x_axis[-1]))
    X_axis = np.arange(1)
    width= 0.2

    plt.bar(X_axis, score1[-1], color='r', width=width, label = 'GMM with vgg16')
    plt.bar(X_axis+0.2, score2[-1], color='b', width=width, label = 'Heirarchical with resnet50')
    plt.bar(X_axis+0.4, score3[-1], color='g', width=width, label = 'Louvain with resnet50')
    plt.bar(X_axis+0.6, score4[-1], color='y', width=width, label = 'Heirarchical with vgg16')

    plt.title("Score comparisons for the best model combinations")
    # plt.xticks(X_axis+0.3, x_axis[-1])
    plt.xlabel('Calinski Harabasz')
    plt.legend()
    plt.show()
    # print(score1)
    # print(score2)
    # print(score3)
    # print(score4)

def apply_algorithms(data, datasets, feature_type):
    kmeans_assignments = {}
    gmm_assignments = {}
    hierarchical_clustering_assignments = {}
    louvain_assignments = {}
    representation_testing_data = {}
    results = {}
    for representation in data_management.representations:
        print("For Representation: ", representation)
        dataset = datasets[representation][feature_type]
        test_data, test_label = data.get_testing_data(dataset, representation)
        representation_testing_data[representation] = test_data, test_label
        # K-means
        kmeans_assignment = apply_kmeans(test_data, test_label)
        # kmeans_assignments[representation] = kmeans_assignment

        # GMM
        gmm_assignment = apply_gmm(test_data, test_label)
        # gmm_assignments[representation] = gmm_assignment

        # Hierarchical Clustering
        hc_assignment = apply_hierarchical_clustering(test_data, test_label)
        # hierarchical_clustering_assignments[representation] = hc_assignment

        # Louvain
        louvain_assignment = apply_louvain(test_data, test_label)
        # louvain_assignments[representation] = louvain_assignment

        results[representation] = {
            "K-Means": kmeans_assignment,
            "GMM": gmm_assignment,
            "Heirarchical Clustering": hc_assignment,
            "Louvain": louvain_assignment

        }
        # location of this return call is exiting the for loop after a single iteration
        # return representation_testing_data, kmeans_assignments, gmm_assignments, hierarchical_clustering_assignments, louvain_assignments,
    return results


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
