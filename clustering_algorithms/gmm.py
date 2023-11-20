from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, v_measure_score


"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from umap import UMAP

def read_data_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as file:
        umap_features = np.array(file['umap_feature'])
        pca_features = np.array(file['pca_feature'])
        filenames = np.array(file['file_name'])
    
    return umap_features, pca_features, filenames

def apply_gmm_and_evaluate(data, algorithm_params):
    gmm = GaussianMixture(**algorithm_params)
    cluster_labels = gmm.fit_predict(data)
    
    silhouette = silhouette_score(data, cluster_labels)
    
    return silhouette, cluster_labels

# Iteration
datasets = ["pathologygan_data.h5", "resnet50_data.h5"]
highest_silhouette_scores = {}

for dataset in datasets:
    file_path = dataset
    umap_features, pca_features, filenames = read_data_from_hdf5(file_path)
    dataset_silhouette_scores = []
    
    for representation, features in zip(["UMAP", "PCA"], [umap_features, pca_features]):
        silhouette, cluster_labels = apply_gmm_and_evaluate(features, algorithm_params)
        
        print(f"Dataset: {dataset}, Representation: {representation}")
        print(f"Silhouette Score: {silhouette}")
        
        dataset_silhouette_scores.append(silhouette)
        
        plt.scatter(features[:, 0], features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
        plt.title(f'Clusters - {dataset}, {representation}')
        plt.show()
    
    highest_silhouette_scores[dataset] = max(dataset_silhouette_scores)

print("Highest Silhouette Scores:")
print(highest_silhouette_scores)


"""


def apply_gmm(test_data, test_label):
    print("GMM")
    parameters = []
    silhouette_scores = []
    v_scores = []

    # components = range from 1 to 20
    covariance = ['full', 'tied', 'diag', 'spherical']
    # max_iter = range from 50 to 200
    # n_init = range from 1 to 10
    init_params = ['kmeans', 'k-means++', 'random', 'random_from_data']
    random_state = 0
    warm_start = [True, False]
    # verbose_interval = range from 1 to 20
    for component in range(1,10):
        for cov in covariance:
            for param in init_params:
                for state in warm_start:
                    try:
                        clustering = GaussianMixture(
                            n_components=component,
                            covariance_type=cov,
                            init_params=param,
                            random_state=random_state,
                            warm_start=state,
                        )
                        labels_ = clustering.fit_predict(test_data)
                        score = silhouette_score(test_data, labels_)
                        v_measure = v_measure_score(test_label, labels_)
                        parameters.append([cov, param, random_state, state])
                        silhouette_scores.append(score)
                        v_scores.append(v_measure)
                    except:
                        pass

    highest_v_score = max(v_scores)
    highest_v_score_index = v_scores.index(highest_v_score)
    v_parms = parameters[highest_v_score_index]

    highest_sil_score = max(silhouette_scores)
    highest_sil_score_index = silhouette_scores.index(highest_sil_score)
    s_parms = parameters[highest_sil_score_index]

    results = {
        "v_score": highest_v_score,
        "v_score_params": v_parms,
        "silhouette_score": highest_sil_score,
        "silhouette_score_params": s_parms
    }
    print("Done GMM")
    return results
