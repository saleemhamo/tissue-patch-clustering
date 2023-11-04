import h5py
import numpy as np
import random

"""


"""

feature_types = ['pca', 'umap']
representations = ['pge', 'resnet50', 'inceptionv3', 'vgg16']


class TissuesData:
    def __init__(self):
        self.data_content = load_data_content()

    def get_representation(self, representation):
        return self.data_content[representation]

    def get_feature(self, feature_type='pca'):  # 'pca' / 'umap'
        if feature_type == 'pca':
            return get_pca_feature(self.data_content)
        return get_umap_feature(self.data_content)

    def get_dataset(self, representation: str, feature_type: str):
        return self.get_feature(feature_type)[representation]

    def get_all_datasets(self):
        datasets = {}
        for representation in representations:
            datasets[representation] = {}

        for representation in representations:
            for feature in feature_types:
                datasets[representation][feature] = self.get_dataset(representation, feature)

        return datasets

    def get_labels(self, representation):
        # tissue type as available ground-truth: labels
        filename = np.squeeze(self.data_content[representation]['file_name'])
        filename = np.array([str(x) for x in filename])
        labels = np.array([x.split('/')[2] for x in filename])
        return labels

    def get_testing_data(self, dataset, representation, k=200):
        representation_labels = self.get_labels(representation)
        random.seed(0)
        selected_index = random.sample(list(np.arange(len(dataset))), k)
        test_data = dataset[selected_index]
        test_label = representation_labels[selected_index]

        return test_data, test_label


def load_data_content():
    pge_path = 'colon_nct_feature/pge_dim_reduced_feature.h5'
    resnet50_path = 'colon_nct_feature/resnet50_dim_reduced_feature.h5'
    inceptionv3_path = 'colon_nct_feature/inceptionv3_dim_reduced_feature.h5'
    vgg16_path = 'colon_nct_feature/vgg16_dim_reduced_feature.h5'

    pge_content = h5py.File(pge_path, mode='r')
    resnet50_content = h5py.File(resnet50_path, mode='r')
    inceptionv3_content = h5py.File(inceptionv3_path, mode='r')
    vgg16_content = h5py.File(vgg16_path, mode='r')

    return {
        'pge': pge_content,
        'resnet50': resnet50_content,
        'inceptionv3': inceptionv3_content,
        'vgg16': vgg16_content
    }


def get_umap_feature(data_content):
    # PCA feature from 4 feature sets: pge_latent, resnet50_latent, inceptionv3_latent, vgg16_latent
    pge_pca_feature = data_content['pge']['umap_feature'][...]
    resnet50_pca_feature = data_content['resnet50']['umap_feature'][...]
    inceptionv3_pca_feature = data_content['inceptionv3']['umap_feature'][...]
    vgg16_pca_feature = data_content['vgg16']['umap_feature'][...]

    return {
        'pge': pge_pca_feature,
        'resnet50': resnet50_pca_feature,
        'inceptionv3': inceptionv3_pca_feature,
        'vgg16': vgg16_pca_feature
    }


def get_pca_feature(data_content):
    # PCA feature from 4 feature sets: pge_latent, resnet50_latent, inceptionv3_latent, vgg16_latent
    pge_pca_feature = data_content['pge']['pca_feature'][...]
    resnet50_pca_feature = data_content['resnet50']['pca_feature'][...]
    inceptionv3_pca_feature = data_content['inceptionv3']['pca_feature'][...]
    vgg16_pca_feature = data_content['vgg16']['pca_feature'][...]

    return {
        'pge': pge_pca_feature,
        'resnet50': resnet50_pca_feature,
        'inceptionv3': inceptionv3_pca_feature,
        'vgg16': vgg16_pca_feature
    }
