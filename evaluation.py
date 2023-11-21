from sklearn.metrics import silhouette_score, v_measure_score, davies_bouldin_score, calinski_harabasz_score

"""
small functions that return the various scores and parameters for the performance of the clustering algorithms
"""
def best_params(scores, parameters):
    return [max(scores), parameters[scores.index(max(scores))]]
def best_db_scores_and_params(scores,parameters):
    return [min(scores), parameters[scores.index(min(scores))]]
def find_silhouette_score(test_data, pred_labels):
    return silhouette_score(test_data, pred_labels)

def find_v_measure(test_label, pred_labels):
    return v_measure_score(test_label, pred_labels)

def find_davies_bouldin_score(test_data, pred_labels):
    return davies_bouldin_score(test_data, pred_labels)

def find_calinski_harabasz_score(test_data, pred_labels):
    return calinski_harabasz_score(test_data, pred_labels)

# I think the following was used in the initial commit
# -------------------------------------------------------------------------------------------------------------
# kmeans_v_measure = v_measure_score(test_label, kmeans_assignment)
# louvain_v_measure = v_measure_score(test_label, louvain_assignment)
# pd.DataFrame({
#     'Metrics': ['silhouette', 'V-measure'],
#     'Kmeans': [kmeans_silhouette, kmeans_v_measure],
#     'Louvain': [louvain_silhouette, louvain_v_measure]
# }).set_index('Metrics')

#
# def calculate_percent(sub_df, attrib):
#     cnt = sub_df[attrib].count()
#     output_sub_df = sub_df.groupby(attrib).count()
#     return (output_sub_df / cnt)
#
#
# resulted_cluster_df = pd.DataFrame({
#     'clusterID': kmeans_assignment,
#     'type': test_label
# })
#
# label_proportion_df = (
#     resulted_cluster_df.groupby(['clusterID'])
#     .apply(lambda x: calculate_percent(x, 'type'))
#     .rename(columns={'clusterID': 'type_occurrence_percentage'})
#     .reset_index()
# )
#
# pivoted_label_proportion_df = pd.pivot_table(
#     label_proportion_df, index='clusterID', columns='type', values='type_occurrence_percentage'
# )
#
# f, axes = plt.subplots(1, 2, figsize=(20, 5))
# number_of_tile_df = (
#     resulted_cluster_df.groupby('clusterID')['type']
#     .count()
#     .reset_index()
#     .rename(columns={'type': 'number_of_tile'})
# )
#
# df_idx = pivoted_label_proportion_df.index
# (pivoted_label_proportion_df * 100).loc[df_idx].plot.bar(stacked=True, ax=axes[0])
#
# axes[0].set_ylabel('Percentage of tissue type')
# axes[0].legend(loc='upper right')
# axes[0].set_title('Cluster configuration by Kmeans')
#
# resulted_cluster_df = pd.DataFrame({
#     'clusterID': louvain_assignment,
#     'type': test_label
# })
#
# label_proportion_df = (
#     resulted_cluster_df.groupby(['clusterID'])
#     .apply(lambda x: calculate_percent(x, 'type'))
#     .rename(columns={'clusterID': 'type_occurrence_percentage'})
#     .reset_index()
# )
#
# pivoted_label_proportion_df = pd.pivot_table(
#     label_proportion_df, index='clusterID', columns='type', values='type_occurrence_percentage'
# )
#
# number_of_tile_df = (
#     resulted_cluster_df.groupby('clusterID')['type']
#     .count()
#     .reset_index()
#     .rename(columns={'type': 'number_of_tile'})
# )
#
# df_idx = pivoted_label_proportion_df.index
# (pivoted_label_proportion_df * 100).loc[df_idx].plot.bar(stacked=True, ax=axes[1])
#
# axes[1].set_ylabel('Percentage of tissue type')
# axes[1].legend(loc='upper right')
# axes[1].set_title('Cluster configuration by Louvain')
# f.show()
