from sklearn.mixture import GaussianMixture

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


def apply_gmm(test_data, n_clusters=9, random_state=0):
    # Fit a Gaussian Mixture Model to the test_data
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm.fit(test_data)

    # Predict cluster assignments for the data
    gmm_assignments = gmm.predict(test_data)
    return gmm_assignments
