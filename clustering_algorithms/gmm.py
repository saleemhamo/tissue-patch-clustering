from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, v_measure_score


"""
import h5py
from sklearn.decomposition import PCA
import umap
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def load_data(file_path, dataset_key):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_key][:]
    return data

def apply_pca(data, n_components=100):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca_data

def apply_umap(data, n_components=100):
    umap_model = umap.UMAP(n_components=n_components)
    umap_data = umap_model.fit_transform(data)
    return umap_data

def perform_gmm_clustering(data, n_components=3):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels

def visualize_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Load Data
pathologygan_data_pca= load_data('pathologygan_data.h5', 'pca_feature')
resnet50_data_pca = load_data('resnet50_data.h5', 'pca_feature')
pathologygan_data_umap= load_data('pathologygan_data.h5', 'umap_feature')
resnet50_data_umap = load_data('resnet50_data.h5', 'umap_feature')

# Apply PCA
pathologygan_pca_data = apply_pca(pathologygan_data_pca)
resnet50_pca_data = apply_pca(resnet50_data_pca)

# Apply UMAP
pathologygan_umap_data = apply_umap(pathologygan_data_umap)
resnet50_umap_data = apply_umap(resnet50_data_umap)

# Perform GMM Clustering
pathologygan_pca_labels = perform_gmm_clustering(pathologygan_pca_data)
pathologygan_umap_labels = perform_gmm_clustering(pathologygan_umap_data)
resnet50_pca_labels = perform_gmm_clustering(resnet50_pca_data)
resnet50_umap_labels = perform_gmm_clustering(resnet50_umap_data)

# Visualize Clusters
visualize_clusters(pathologygan_pca_data, pathologygan_pca_labels, 'PathologyGAN PCA Clustering')
visualize_clusters(pathologygan_umap_data, pathologygan_umap_labels, 'PathologyGAN UMAP Clustering')
visualize_clusters(resnet50_pca_data, resnet50_pca_labels, 'ResNet50 PCA Clustering')
visualize_clusters(resnet50_umap_data, resnet50_umap_labels, 'ResNet50 UMAP Clustering')



"""


def apply_gmm(test_data, test_label):
    parameters = []
    silhouette_scores = []
    v_scores = []

    # components = range from 1 to 20
    covariance = ['full', 'tied', 'diag', 'spherical']
    tol = 0.0001    #float from 0.0001 to 0.01
    reg_covar = 0.0000001 #float from 0.0000001 to 0.00001
    # max_iter = range from 50 to 200
    # n_init = range from 1 to 10
    init_params = ['kmeans', 'k-means++', 'random', 'random_from_data']
    random_state = 0
    warm_start = [True, False]
    # verbose_interval = range from 1 to 20
    for component in range(1,20):
        for cov in covariance:
            while tol < 0.01:
                while reg_covar < 0.00001:
                    for max_iter in range(50,200):
                        for n_init in range(1,20):
                            for param in init_params:
                                for state in warm_start:
                                    for verbose_interval in range(1,20):
                                        try:
                                            clustering = GaussianMixture(
                                                n_components=component,
                                                covariance_type=cov,
                                                tol=tol,
                                                reg_covar=reg_covar,
                                                max_iter=max_iter,
                                                n_init=n_init,
                                                init_params=param,
                                                random_state=random_state,
                                                warm_start=state,
                                                verbose_interval=verbose_interval
                                            )
                                            labels_ = clustering.fit_predict(test_data)
                                            score = silhouette_score(test_data, labels_)
                                            v_measure = v_measure_score(test_label, labels_)
                                            parameters.append([cov, tol, reg_covar, max_iter, n_init, param, random_state, state, verbose_interval])
                                            silhouette_scores.append(score)
                                            v_scores.append(v_measure)
                                        except:
                                            pass
                    reg_covar += 0.0000001
                tol += 0.0001

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
    return results
