from sklearn.mixture import GaussianMixture

"""


"""


def apply_gmm(test_data, n_clusters=9, random_state=0):
    # Fit a Gaussian Mixture Model to the test_data
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm.fit(test_data)

    # Predict cluster assignments for the data
    gmm_assignments = gmm.predict(test_data)
    return gmm_assignments
