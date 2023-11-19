from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sknetwork.clustering import Louvain  # search for scikit-network when trying to install
from sklearn.metrics import silhouette_score, v_measure_score

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
    while res < 10:
        for ver in verbosity:
            for mod in modularity_options:
                try:
                    clustering = Louvain(
                        resolution=res,
                        modularity=mod,
                        verbose=ver,
                        random_state=0)
                    labels_ = clustering.fit_predict(test_data)
                    score = silhouette_score(test_data, labels_)
                    v_measure = v_measure_score(test_label, labels_)
                    parameters.append([res, ver, mod])
                    silhouette_scores.append(score)
                    v_scores.append(v_measure)

                except:
                    pass
        res += 0.1

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
