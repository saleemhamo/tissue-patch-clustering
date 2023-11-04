from sklearn.cluster import KMeans

"""


"""


def apply_kmeans(test_data, n_clusters=3, random_state=0):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans_assignment = kmeans_model.fit_predict(test_data)
    return kmeans_assignment
