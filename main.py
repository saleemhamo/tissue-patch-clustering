import numpy as np
import plotly.graph_objects as go
from sknetwork.clustering import Louvain
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import peformance_evaluation
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
    # # a. PCA
    pca_assignments = apply_algorithms(data, datasets, 'pca')
    # # b. UMAP
    umap_assignments = apply_algorithms(data, datasets, 'umap')
    peformance_evaluation.make_plots(pca_assignments, 'pca')
    peformance_evaluation.make_plots(umap_assignments, 'pca')

    evaluate(datasets, data)
    """The parameters for the best metric score were retrieved manually and saved to the lists that are defined at the
    top of the evaluate function. Each set of parameters is used to make a new model with the appropriate configurations
    and the performance metric results are graphed against each other.
    """
def evaluate(datasets, data):
    # The top evaluated model configurations with their respective best parameters
    gmm_umap_vgg16_params = [9, 'spherical', 'random', 0, True] # GMM umap vgg16
    hc_umap_resnet50_params = [9, 'euclidean', 'auto', 'complete'] # Heirarchical CLustering, UMAP, resnet50
    louvain_umap_resnet50_params = [2.3000000000000007, True, 'Potts'] #Louvain, UMAP, resnet50
    hc_umap_vgg16_params = [9, 'euclidean', 'auto', 'ward'] # Heirarchical Clustering, UMAP, vgg16


    # recreating the data sets to make the models
    v_dataset = datasets['vgg16']['umap']
    sil_dataset = datasets['resnet50']['umap']
    db_dataset = datasets['resnet50']['umap']
    ch_dataset = datasets['vgg16']['umap']

    # model and scores for the best v_score from the testing which is Gaussian Mixture with Umap and vgg
    # building the 200 line sample test data, making the model with optimal parameters and evaluating the performance
    # with v score, silhouette score, davies bouldin, and calinski harabasz
    test_data, test_label = data.get_testing_data(v_dataset, 'vgg16')
    clustering = GaussianMixture(
                        n_components=gmm_umap_vgg16_params[0],
                        covariance_type=gmm_umap_vgg16_params[1],
                        init_params=gmm_umap_vgg16_params[2],
                        random_state=gmm_umap_vgg16_params[3],
                        warm_start=gmm_umap_vgg16_params[4],
                    )
    pred_labels = clustering.fit_predict(test_data)
    vgg_gmm_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    # model and scores for the best silhouette_score from the testing which is Heirarchical Clusterig with Umap and resnet50
    # building the 200 line sample test data, making the model with optimal parameters and evaluating the performance
    # with v score, silhouette score, davies bouldin, and calinski harabasz
    test_data, test_label = data.get_testing_data(sil_dataset, 'resnet50')
    clustering = AgglomerativeClustering(
                        n_clusters=hc_umap_resnet50_params[0],
                        metric=hc_umap_resnet50_params[1],
                        compute_full_tree=hc_umap_resnet50_params[2],
                        linkage=hc_umap_resnet50_params[3])
    pred_labels = clustering.fit_predict(test_data)
    resnet_hc_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    # model and scores for the best davies bouldin from the testing which is Louvain with Umap and resnet50
    # building the 200 line sample test data, making the model with optimal parameters and evaluating the performance
    # with v score, silhouette score, davies bouldin, and calinski harabasz
    test_data, test_label = data.get_testing_data(db_dataset, 'resnet50')
    clustering = Louvain(
                        resolution=louvain_umap_resnet50_params[0],
                        modularity=louvain_umap_resnet50_params[2],
                        verbose=louvain_umap_resnet50_params[1],
                        random_state=0)
    pred_labels = clustering.fit_predict(test_data)
    resnet_louv_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    # model and scores for the best calinski harabasz from the testing which is Heirarchical Clusterig with Umap and vgg16
    # building the 200 line sample test data, making the model with optimal parameters and evaluating the performance
    # with v score, silhouette score, davies bouldin, and calinski harabasz
    test_data, test_label = data.get_testing_data(ch_dataset, 'vgg16')
    clustering = AgglomerativeClustering(
                        n_clusters=hc_umap_vgg16_params[0],
                        metric=hc_umap_vgg16_params[1],
                        compute_full_tree=hc_umap_vgg16_params[2],
                        linkage=hc_umap_vgg16_params[3])
    pred_labels = clustering.fit_predict(test_data)
    vgg_hc_scores = [v_measure_score(test_label, pred_labels), silhouette_score(test_data, pred_labels),davies_bouldin_score(test_data, pred_labels),calinski_harabasz_score(test_data, pred_labels)]

    # All the performance metrics are plotted against each other
    plot_results(vgg_gmm_scores, resnet_hc_scores, resnet_louv_scores,vgg_hc_scores)
def plot_results(score1, score2, score3, score4):
    # function inputs are 4 lists of scoress in the order of vscore, sil score, daview bouldin, calsinski harabsz
    x_axis = ['V_score', 'Silhouette Score', 'Daview Bouldin', 'Calinski Harabasz']
    X_axis = np.arange(len(x_axis[:-1]))
    width= 0.2

    # Plotting the first three metrics since their scale is more similar 
    plt.bar(X_axis, score1[:-1], color='r', width=width, label = 'GMM with vgg16')
    plt.bar(X_axis+0.2, score2[:-1], color='b', width=width, label = 'Heirarchical with resnet50')
    plt.bar(X_axis+0.4, score3[:-1], color='g', width=width, label = 'Louvain with resnet50')
    plt.bar(X_axis+0.6, score4[:-1], color='y', width=width, label = 'Heirarchical with vgg16')

    plt.title("Score comparisons for the best model combinations")
    plt.xticks(X_axis+0.3, x_axis[:-1])
    plt.xlabel('Calinski Harabasz')
    plt.legend()
    plt.show()

    # plots the Calinski Harabasz score since its scale is much larger than the other metrics
    plt.bar(X_axis, score1[-1], color='r', width=width, label = 'GMM with vgg16')
    plt.bar(X_axis+0.2, score2[-1], color='b', width=width, label = 'Heirarchical with resnet50')
    plt.bar(X_axis+0.4, score3[-1], color='g', width=width, label = 'Louvain with resnet50')
    plt.bar(X_axis+0.6, score4[-1], color='y', width=width, label = 'Heirarchical with vgg16')

    plt.title("Score comparisons for the best model combinations")
    # plt.xticks(X_axis+0.3, x_axis[-1])
    plt.xlabel('Calinski Harabasz')
    plt.legend()
    plt.show()

def apply_algorithms(data, datasets, feature_type):
    results = {}
    for representation in data_management.representations:
        print("For Representation: ", representation)
        dataset = datasets[representation][feature_type]
        test_data, test_label = data.get_testing_data(dataset, representation)

        # K-means
        kmeans_assignment = apply_kmeans(test_data, test_label)

        # GMM
        gmm_assignment = apply_gmm(test_data, test_label)

        # Hierarchical Clustering
        hc_assignment = apply_hierarchical_clustering(test_data, test_label)

        # Louvain
        louvain_assignment = apply_louvain(test_data, test_label)

        results[representation] = {
            "K-Means": kmeans_assignment,
            "GMM": gmm_assignment,
            "Heirarchical Clustering": hc_assignment,
            "Louvain": louvain_assignment

        }
    return results

# Early stage function used to visulaize the data before starting on creating the algorithms
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

