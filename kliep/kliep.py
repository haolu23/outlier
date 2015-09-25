import numpy as np
import cvxpy as cvx
from scipy import spatial


def kliep(phi_te, phi_tr):
    """
    using algorithm in the paper to estimate alpha
    same as matlab code
    """
    b = phi_tr.mean(axis=0)
    init_a = np.ones(phi_te.shape[1])
    a = init_a
    c = b.dot(b)
    for epsilon in 1000**np.arange(3, -4, -1):
        while True:
            old_score = phi_te.dot(a).mean()
            a = a + epsilon * phi_te.T.dot(1/phi_te.dot(a))
            a = a + (1-b.dot(a))*b/c
            a = np.maximum(a, 0)
            a = a / b.dot(a)
            score = np.log(phi_te.dot(a)).mean()
            if score <= old_score:
                break
    return a


def kliep_learning(phi_te, phi_tr):
    """
    solve the convex optimization problem using cvxpy
    phi_te is kernel of test samples
    phi_tr is kernel of training samples
    """
    x = cvx.Variable(phi_te.shape[1])
    objective = cvx.Maximize(cvx.sum_entries(cvx.log(phi_te*x)))
    constraints = [cvx.sum_entries(phi_tr*x) == phi_tr.shape[0], x>=0]
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    return prob, x


class Kliep(object):
    def __init__(self, width, lower=100):
        self.nlower = lower
        self.kernel_width = width

    def fit(self, x_train, x_test):
        # decide random Gaussian centers
        nkernel = min(self.nlower, x_test.shape[0])
        kernels = x_test[np.random.choice(np.arange(x_test.shape[0]), size=nkernel, replace=False), :]
        phi_te = np.exp(-spatial.distance.cdist(x_test, kernels)**2/self.kernel_width**2/2.0)
        phi_tr = np.exp(-spatial.distance.cdist(x_train, kernels)**2/self.kernel_width**2/2.0)
        self.kernel_ = kernels
        # too slow to converge
        #prob, self.x_ = kliep_learning(phi_te, phi_tr)
        #self.x_ = self.x_.value
        #self.score_ = prob.value / x_test.shape[0]

        self.x_ = kliep(phi_te, phi_tr)
        self.score_ = phi_te.dot(self.x_).mean()
        return self

    def predict(self, x_train):
        phi_tr = np.exp(-spatial.distance.cdist(x_train, self.kernel_)/self.kernel_width**2/2.0)
        return phi_tr.dot(self.x_.value)
