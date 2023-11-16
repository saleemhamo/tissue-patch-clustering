from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, v_measure_score


def apply_kmeans(test_data, test_label):

    initializer = ['k-means++', 'random']
    # clusters: range from 1 to 20
    # n_init 'auto'
    # max_iter int between 1 and 999
    tol = 0.00001 # tol float between 0 and 0.001

    # verbose int --> pretty much a random seed = 0
    # random_state = 0
    copy_x = [True, False]
    algorithm = 'lloyd'
    verbose = 0
    random_state = 0
    n_init = 'auto'
    parameters = []
    silhouette_scores = []
    v_scores = []

    for cluster in range(1,20):
        for iterations in range(150, 400):
            while tol < 0.001:
                for copy in copy_x:
                    for init in initializer:
                        try:
                            clustering = KMeans(
                                init=init,
                                n_clusters=cluster,
                                n_init=n_init,
                                max_iter=iterations,
                                tol=tol,
                                verbose=verbose,
                                random_state=random_state,
                                copy_x=copy,
                                algorithm=algorithm
                            )
                            labels_ = clustering.fit_predict(test_data)
                            score = silhouette_score(test_data, labels_)
                            v_measure = v_measure_score(test_label, labels_)

                            parameters.append([init, cluster, n_init, iterations,tol, verbose, random_state])

                            silhouette_scores.append(score)
                            v_scores.append(v_measure)

                        except:
                            pass
                tol += 0.00001

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
