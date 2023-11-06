from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sknetwork.clustering import Louvain    #search for scikit-network when trying to install

"""


"""

def apply_louvain(test_data, resolution=0.9, modularity='Newman', random_state=0):
    louvain_model = Louvain(resolution=resolution, modularity=modularity, random_state=random_state)
    adjacency_matrix = sparse.csr_matrix(MinMaxScaler().fit_transform(-pairwise_distances(test_data)))
    louvain_assignment = louvain_model.fit_transform(adjacency_matrix)
    return louvain_assignment
