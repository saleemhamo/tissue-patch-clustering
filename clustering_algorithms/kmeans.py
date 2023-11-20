from sklearn.cluster import KMeans
import evaluation

def apply_kmeans(test_data, test_label):
    print("Kmeans")
    parameters = []
    silhouette_scores = []
    v_scores = []
    db_scores = []
    ch_scores = []
    init = ['k-means++','random']
    copy_x = [True, False]
    algorithm = ['lloyd', 'elkan']
    k = 9
    for init_ in init:
        for copy in copy_x:
            for alg in algorithm:
                pred_labels = KMeans(n_clusters=k,
                                    random_state=42,
                                    n_init='auto',
                                    init= init_,
                                    copy_x=copy,
                                    algorithm=alg).fit_predict(test_data)

                silhouette_scores.append(evaluation.silhouette_score(test_data, pred_labels))
                v_scores.append(evaluation.v_measure_score(test_label, pred_labels))
                ch_scores.append(evaluation.find_calinski_harabasz_score(test_data, pred_labels))
                db_scores.append(evaluation.find_davies_bouldin_score(test_data, pred_labels))

                parameters.append([k, 42, 'auto', init_, copy, alg])

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

    # cv_scores = []
    # v_scores = []
    # f1_scores = []  # Perhaps not to be considered
    # silhouette_scores = []
    #
    # # Testing for different cluster values
    # k_values = list(range(1, 41))
    # for k in k_values:
    #     X_train, X_test, y_train, y_test = train_test_split(test_data, test_label, test_size=.1, random_state=4)
    #
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(X_train)
    #     y_pred = kmeans.predict(X_test)
    #
    #     # Deplying k-nearest neighbors classifier
    #     # knn = KNeighborsClassifier(n_neighbors=k)
    #     # knn.fit(X_train, y_train)
    #     # y_pred_knn = knn.predict(X_test)
    #
    #
    #     # Calculating the validation scores. Notice that the number of folds can be increased if required
    #     # cv_score = np.mean(cross_val_score(kmeans, test_data, test_label, cv=7))
    #     v_score = v_measure_score(y_test, y_pred)
    #     # f1 = f1_score(test_label, y_pred_knn)
    #     silhouette = silhouette_score(X_test, y_pred)
    #
    #     # Appending the scores to the previous list
    #     # cv_scores.append(cv_score)
    #     v_scores.append(v_score)
    #     # f1_scores.append(f1)
    #     silhouette_scores.append(silhouette)



    # best_k = k_values[np.argmax(cv_scores)]
    # print(f'The best k value is {best_k}.')



