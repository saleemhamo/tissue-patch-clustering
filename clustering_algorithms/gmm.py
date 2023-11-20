from sklearn.mixture import GaussianMixture
import evaluation


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
    db_scores = []
    ch_scores = []

    # components = range from 1 to 20
    covariance = ['full', 'tied', 'diag', 'spherical']
    init_params = ['kmeans', 'k-means++', 'random', 'random_from_data']
    random_state = 0
    warm_start = [True, False]

    for component in range(1,20):
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
                        pred_labels = clustering.fit_predict(test_data)

                        db_scores.append(evaluation.find_davies_bouldin_score(test_data, pred_labels))
                        silhouette_scores.append(evaluation.silhouette_score(test_data, pred_labels))
                        v_scores.append(evaluation.v_measure_score(test_label, pred_labels))
                        ch_scores.append(evaluation.find_calinski_harabasz_score(test_data, pred_labels))

                        parameters.append([cov, param, random_state, state])

                    except:
                        pass

    v_score_and_params = evaluation.best_params(v_scores, parameters)
    sil_scores_and_params = evaluation.best_params(silhouette_scores, parameters)
    db_scores_and_params = evaluation.best_db_scores_and_params(db_scores, parameters)
    ch_scores_and_params = evaluation.best_params(ch_scores, parameters)

    results = {
        "v_score": v_score_and_params[0],
        "v_score_params": v_score_and_params[1],
        "silhouette_score": sil_scores_and_params[0],
        "silhouette_score_params": sil_scores_and_params[1],
        "davies_bouldin_score": db_scores_and_params[0],
        "davies_bouldin_params": db_scores_and_params[1],
        "calinski_harabasz_score": ch_scores_and_params[0],
        "calinski_harabasz_params": ch_scores_and_params[1]
    }
    return results
