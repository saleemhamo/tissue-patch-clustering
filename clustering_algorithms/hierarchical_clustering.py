import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score,v_measure_score
import matplotlib.pyplot as plt

"""


"""


def apply_hierarchical_clustering(test_data, test_label):
    # You have been given sampled test data of 200x 100 item vector --> brought down from 5000x100
    # As well as the associated labels for those pieces of data
    clusters = len(np.unique(test_label))
    metrics_ = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    # won't use memory
    compute_full_tree = ['auto', 'bool']
    linkages = ['ward', 'complete', 'average', 'single']
    # since we have a known number of cluster, distance_threashold will not be used

    # scan = DBSCAN(eps=3, min_samples=5)
    # scan_labels = scan.fit_predict(x_train)
    # db_score = silhouette_score(x_train, scan_labels)
    # print("Scan Score:", db_score)

    # v measure or sillohette score

    parameters = []
    silhouette_scores = []
    v_scores = []
    for i in range(2, clusters+1):
        for metric in metrics_:
            for comp in compute_full_tree:
                for link in linkages:
                    try:
                        clustering = AgglomerativeClustering(
                            n_clusters=i,
                            metric=metric,
                            compute_full_tree=comp,
                            linkage=link)
                        labels_ = clustering.fit_predict(test_data)
                        score = silhouette_score(test_data, labels_)
                        v_measure = v_measure_score(test_label, labels_)
                        parameters.append([i,metric,comp,link])
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

    print("The best v-score was: ", highest_v_score, " with parameters:")
    print(v_parms)

    print("The best silhouette-score was: ", highest_sil_score, " with parameters:")
    print(s_parms)


    # Access the cluster labels for each data point
    # cluster_labels = clustering.labels_

    # Return the performance metrics as a dictionary
    # return cluster_labels
