import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import pdb

class Lof():
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        #tree = sp.spatial.KDTree(data)
        distance = squareform(pdist(data))
        indices = stats.mstats.rankdata(distance, axis=1)

        indices_k = indices <= self.k
        #pdb.set_trace()
        # k distance
        kdist = np.zeros(len(data))
        for i in range(data.shape[0]):
            kneighbours = distance[i, indices_k[i, :]]
            kdist[i] = kneighbours.max()
        
        lrd = np.zeros(len(data))
        for i in range(data.shape[0]):
            # reachability distance of k nearest points
            # lrd
            lrd[i] = 1/np.maximum(kdist[indices_k[i, :]], distance[i, indices_k[i, :]]).mean()
        # lof
        #pdb.set_trace()
        lof = np.zeros(len(data))
        for i in range(data.shape[0]):
            lof[i] = lrd[indices_k[i, :]].mean()/lrd[i]
        return lof

if __name__ == "__main__":
    x = np.vstack((np.random.random((400, 2)), 100*np.random.random((3, 2))))
    m = Lof(10)
    m.fit(x)
