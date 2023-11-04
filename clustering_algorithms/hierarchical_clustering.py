from sklearn.cluster import AgglomerativeClustering

"""


"""


def apply_hierarchical_clustering(test_data, n_clusters=9):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(test_data)

    # Access the cluster labels for each data point
    cluster_labels = clustering.labels_
    print(len(cluster_labels))
    return cluster_labels
