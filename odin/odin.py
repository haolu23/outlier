import numpy as np
import networkx as nx
from scipy.spatial import KDTree
#import pdb
#
#
# implement algorithms in "Outlier Dection Using k-Nearest Neighbour Graph"
#
#

class KnnGraph(object):
    def __init__(self, k):
        self.k = k

    def fit(self, x):
        tree = KDTree(x)
        self.knngraph = nx.DiGraph()

        self.knngraph.add_nodes_from(range(len(x)))
        for i in range(len(x)):
            neighbors_distance, neighbors_index = tree.query(x[i, :], self.k)
            self.knngraph.add_weighted_edges_from(
                zip([i]*self.k, neighbors_index, neighbors_distance))


class Odin(KnnGraph):
    def __init__(self, k, t):
        super(self.__class__, self).__init__(k)
        self.t = t


    def fit(self, x):
        super(self.__class__, self).fit(x)
        return np.array([i for i in range(len(x)) if self.knngraph.in_degree(i) < self.t])

    
class Meandist(KnnGraph):
    def __init__(self, k, t):
        super(self.__class__, self).__init__(k)
        self.t = t

    def fit(self, x):
        super(self.__class__, self).fit(x)
        dist = np.array([self.knngraph.degree(i, weight="weight") for i in range(len(x))])
        ind = np.argsort(dist)
        meandistdiff = np.diff(dist[ind])
        #TODO: in case there is no index satisfying the threshold condition
        firstexceed = np.argmax(meandistdiff > meandistdiff.max() * self.t)
        return np.arange(firstexceed, len(x))

