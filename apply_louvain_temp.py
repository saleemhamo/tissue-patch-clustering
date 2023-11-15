import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data
from clustering_algorithms.louvain import apply_louvain
from data import TissuesData
from evaluation import find_silhouette_score, find_v_measure

feature_types = ['pca', 'umap']
representations = ['pge', 'resnet50', 'inceptionv3', 'vgg16']


def main():
    """ 1. Load dataset """
    tissues_data = TissuesData()  # Load data in constructor

    """ 2. Get all sample combinations (Feature types *  Data Representation) """
    datasets = tissues_data.get_all_datasets()

    results_headers = dict({'Metrics': ['count', 'silhouette', 'V-measure']})

    modularity_options = ['Dugue', 'Newman', 'Potts']
    random_state = 0
    for resolution in [0.9, 1, 0.8]:
        for feature_type in feature_types:  # 'pca' or 'umap'
            for modularity in modularity_options:
                try:
                    louvain_results = apply_evaluation_for_louvain(
                        tissues_data, datasets, feature_type, resolution, modularity, random_state
                    )

                    louvain_results_dictionary = results_headers.copy()
                    louvain_results_dictionary.update(louvain_results)

                    frame = pd.DataFrame(louvain_results_dictionary).set_index('Metrics')
                    fig = plt.figure(figsize=(8, 2))
                    ax = fig.add_subplot(111)
                    ax.table(cellText=frame.values, rowLabels=frame.index, colLabels=frame.columns, loc="center")
                    ax.set_title("Louvain Results (" + feature_type + "): Resolution=" + str(
                        resolution) + ", Modularity=" + modularity + ", Random State=" + str(random_state))
                    ax.axis("off")
                    fig.show()
                except:
                    print("Error in: (" + feature_type + "): Resolution=" + str(
                        resolution) + ", Modularity=" + modularity + ", Random State=" + str(random_state))


def apply_evaluation_for_louvain(tissues_data, datasets, feature_type, resolution=0.9, modularity='Newman',
                                 random_state=0):
    louvain_results = {}
    for representation in representations:
        louvain_results[representation] = []
        dataset = datasets[representation][feature_type]
        test_data, test_label = tissues_data.get_testing_data(dataset, representation)

        louvain_assignment, labels = apply_louvain(test_data, resolution, modularity, random_state)

        counts = np.unique(labels, return_counts=True)
        louvain_results[representation].append(int(counts[0].size))
        silhouette = find_silhouette_score(test_data, labels)
        louvain_results[representation].append(silhouette)
        v_measure = find_v_measure(test_label, labels)
        louvain_results[representation].append(v_measure)
    return louvain_results


if __name__ == '__main__':
    main()
