import numpy as np
from scipy.spatial import KDTree
cimport numpy as np

INT = np.int
DOUBLE = np.float64
ctypedef np.int_t INT_t
ctypedef np.float64_t DOUBLE_t

class Cof():
    def __init__(self, int k):
        self.k = k

    def fit(self, np.ndarray[DOUBLE_t, ndim=2] x):
        tree = KDTree(x)
        # find k nearest neighbours for all points
        cdef np.ndarray[INT_t, ndim=2] neighbors_index = np.zeros((len(x), self.k), dtype=INT)
        cdef np.ndarray[DOUBLE_t, ndim=2] neighbors_distance = np.zeros_like(neighbors_index, dtype=DOUBLE)
        cdef int i, j, k, point_index
        
        for i in range(len(x)):
            neighbors_distance[i, :], neighbors_index[i, :] = tree.query(x[i, :], self.k)

        # calculate SBN-path
        cdef np.ndarray[DOUBLE_t, ndim=1] sbnpath = np.zeros(len(x), dtype=DOUBLE)

        cdef double shortest_distance
        cdef int next_point
        cdef double path_distance
        for i in range(len(x)):
            path = set()
            path.add(i)
            path_distance = 0.0
            
            for j in range(self.k):
                shortest_distance = neighbors_distance[i, -1] * 10
                next_point = 0
                for point_index in path:
                    k = 0
                    while k < self.k and neighbors_index[point_index, k] in path:
                        k = k+1
                    if k < self.k and shortest_distance > neighbors_distance[point_index, k]:
                        shortest_distance = neighbors_distance[point_index, k]
                        next_point = neighbors_index[point_index, k]
                        
                path.add(next_point)
                
                path_distance = path_distance + shortest_distance * 2.0 * (self.k - j)
                
            sbnpath[i] = path_distance / self.k / (self.k + 1)

        # calculate cof
        return sbnpath / sbnpath[neighbors_index].mean(axis=1)
