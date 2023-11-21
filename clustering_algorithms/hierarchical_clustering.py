import numpy as np
from sklearn.cluster import AgglomerativeClustering
import evaluation


def apply_hierarchical_clustering(test_data, test_label):
    print("HC")

    # You have been given sampled test data of 200x 100 item vector --> brought down from 5000x100
    # As well as the associated labels for those pieces of data
    clusters = len(np.unique(test_label))
    metrics_ = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    compute_full_tree = ['auto', 'bool']
    linkages = ['ward', 'complete', 'average', 'single']
    # since we have a known number of cluster, distance_threashold will not be used

    parameters = []
    silhouette_scores = []
    v_scores = []
    db_scores = []
    ch_scores = []
    i = 9
    for metric in metrics_:
        for comp in compute_full_tree:
            for link in linkages:
                # some combinations of parameters return an error, the try except skips those combinations
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=i,
                        metric=metric,
                        compute_full_tree=comp,
                        linkage=link)
                    pred_labels = clustering.fit_predict(test_data)

                    # calls performance metrics from evaluation.py
                    parameters.append([i, metric, comp, link])
                    db_scores.append(evaluation.find_davies_bouldin_score(test_data, pred_labels))
                    silhouette_scores.append(evaluation.silhouette_score(test_data, pred_labels))
                    v_scores.append(evaluation.v_measure_score(test_label, pred_labels))
                    ch_scores.append(evaluation.find_calinski_harabasz_score(test_data, pred_labels))

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

