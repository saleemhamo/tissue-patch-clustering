from sknetwork.clustering import Louvain  # search for scikit-network when trying to install
import evaluation

"""


"""


def apply_louvain(test_data, test_label):
    print("Louvain")

    res = 0.1
    modularity_options = ['Dugue', 'Newman', 'Potts']
    verbosity = [True, False]
    random_state = 0
    parameters = []
    silhouette_scores = []
    v_scores = []
    db_scores = []
    ch_scores = []

    while res < 10:
        for ver in verbosity:
            for mod in modularity_options:
                try:
                    clustering = Louvain(
                        resolution=res,
                        modularity=mod,
                        verbose=ver,
                        random_state=0)
                    pred_labels = clustering.fit_predict(test_data)

                    db_scores.append(evaluation.find_davies_bouldin_score(test_data, pred_labels))
                    silhouette_scores.append(evaluation.silhouette_score(test_data, pred_labels))
                    v_scores.append(evaluation.v_measure_score(test_label, pred_labels))
                    ch_scores.append(evaluation.find_calinski_harabasz_score(test_data, pred_labels))

                    parameters.append([res, ver, mod])

                except:
                    pass
        res += 0.1

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
