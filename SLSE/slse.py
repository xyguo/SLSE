# -*- coding: utf-8 -*-
"""
Implementation for the main algorithm of paper

`Scalable Estimating Stochastic Linear Combination of Non-linear Regressions`
Author: Di Wang, Xiangyu Guo, Shi Li, Jinhui Xu

Code Author: Xiangyu Guo
"""

import numpy as np
from sklearn.exceptions import NotFittedError
from scipy.optimize import newton


class SLSE(object):
    """Scaled Least Squared Estimators

    The response y is assumed to be generated via the following model:

    y = \sum^k_{i=1} z_i f_i(\langle \beta^*_i, x \rangle) + \epsilon

    The goal is to estimate {\beta^*_i}_{i\in[k]}

    :param cov_option: 1 or 2, ways to estimate the scaled "covariance".
    :param n_iters: int, maximum number of iterations
    :param tol: double, error tolerance in estimating the scaling constant c
    :param groundtruth: array of shape=(n_links, n_features), true value of {\beta^*_i}_{i\in[k]}
    :param record_all_loss:
    """
    def __init__(self, cov_option=1, n_iters=100, tol=1e-3,
                 groundtruth=None, record_all_loss=False):
        self.cov_option_ = cov_option
        if cov_option not in [1, 2]:
            raise ValueError("Parameter `cov_option` must be either 1 or 2.")
        self.beta_nlr_ = None
        self.eta_ = tol
        self.n_iters_ = n_iters
        self.record_loss_ = record_all_loss
        self.groundtruth_ = groundtruth

    @property
    def estimator(self):
        if self.beta_nlr_ is None:
            raise NotFittedError("Model hasn't been fitted")
        return self.beta_nlr_

    def loss(self, groundtruth):
        """Compare the estimated beta v.s. groundtruth beta

        :param groundtruth: array of shape=(n_links, n_features), true value of {\beta^*_i}_{i\in[k]}
        :return err: array of shape=(n_links,), l2-error for each beta^*_i
        """
        if self.beta_nlr_ is None:
            raise NotFittedError("Model hasn't been fitted")
        return np.linalg.norm(self.beta_nlr_ - groundtruth, axis=1)

    def fit(self, X, Y, Z, F):
        """Fit the model on given input data.

        :param X: array of shape=(n_samples, n_features), feature vector (variates) of training data
        :param Y: array of shape=(n_samples,), objective (response) of training data
        :param Z: array of shape=(n_samples, n_links), random coefficient for combining the link functions
        :param F: list of len=n_links, of which each element f is an object offering the two following methods:
            1. f.eval(x) -> real, which evaluate f at real number x
            2. f.grad(x) -> real, which evaluate f's derivative at real number x
            3. f.ggrad(x) -> real, which evaluate f's 2nd-order derivative at real number x
        :return self:
        """
        n_samples, n_features = X.shape
        _, n_links = Z.shape
        assert n_links == len(F)

        # compute the covariance matrix of the input
        if self.cov_option_ == 1:
            Sigma_inv = np.linalg.inv(X.T.dot(X))
        else:
            Sigma_inv = _sub_sampling_cov(X, s=n_samples / 10)

        # compute the ordinary least squares estimator beta_ols
        YZ = Z * Y.reshape(n_samples, 1)
        beta_ols = Sigma_inv.dot(X.T.dot(YZ))  # beta_ols.shape = (n_features, n_links)
        Y_ols = X.dot(beta_ols).T  # Y_ols.shape = (n_links, n_samples)
        VFprime = [np.vectorize(f.grad) for f in F]
        VFpprime = [np.vectorize(f.ggrad) for f in F]
        L = [lambda c: ((c / n_samples) * np.sum(VFprime[j](Y_ols[j] * c)) - 1)
             for j in range(n_links)]
        L_prime = [lambda c: ((1 / n_samples) * np.sum(VFprime[j](Y_ols[j] * c) +
                                                       c * Y_ols[j] * VFpprime[j](Y_ols[j] * c)))
                   for j in range(n_links)]

        C = [newton(L[j], x0=0, fprime=L_prime[j]) for j in range(n_links)]

        # compute the scale constant for beta_nlr
        self.beta_nlr_ = beta_ols.T * C  # beta_nlr.shape=(n_links, n_features)

        return self


def _sub_sampling_cov(X, s):
    """
    :param X: array of shape=(n_samples, n_features)
    :param s: int, size of the sub-sample
    :return subs_cov: array of shape=(n_features, n_features), the scaled covariance obtained on sub-samples
    """
    n_samples, _ = X.shape
    assert s <= n_samples

    S = np.arange(n_samples)
    np.random.shuffle(S)
    X_S = X[S[:s]]
    subs_cov = np.linalg.inv(X_S.T.dot(X_S)) * s / n_samples

    return subs_cov