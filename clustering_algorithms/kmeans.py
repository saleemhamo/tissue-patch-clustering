from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, silhouette_score, v_measure_score
from sklearn.model_selection import train_test_split, cross_val_score

def apply_kmeans(test_data, test_label):
    print("Kmeans")
    k = 9
    X_train, X_test, y_train, y_test = train_test_split(test_data, test_label, test_size=.1, random_state=4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_train)
    pred = kmeans.predict(X_test)
    v_score = v_measure_score(y_test, pred)
    sil_score = silhouette_score(X_test, y_test)




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


    # highest_v_score = max(v_scores)
    v_parms = [k]

    # highest_sil_score = max(silhouette_scores)
    # highest_sil_score_index = silhouette_scores.index(highest_sil_score)
    s_parms = [k]

    results = {
        "v_score": v_score,
        "v_score_params": v_parms,
        "silhouette_score": sil_score,
        "silhouette_score_params": s_parms
    }
    print('done KMeans')
    return results
