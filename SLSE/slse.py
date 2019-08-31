# -*- coding: utf-8 -*-
"""
Implementation for the main algorithm of paper

`Scalable Estimating Stochastic Linear Combination of Non-linear Regressions`
Author: Di Wang, Xiangyu Guo, Shi Li, Jinhui Xu

Code Author: Xiangyu Guo
"""

import numpy as np
from sklearn.exceptions import NotFittedError
from scipy.optimize import newton, brentq
import warnings


class SLSE(object):
    """Scaled Least Squared Estimators

    The response y is assumed to be generated via the following model:

    y = \sum^k_{i=1} z_i f_i(\langle \beta^*_i, x \rangle) + \epsilon

    The goal is to estimate {\beta^*_i}_{i\in[k]}

    :param cov_option: 1 or 2, ways to estimate the scaled "covariance".
    :param n_iters: int, maximum number of iterations
    :param tol: double, error tolerance in estimating the scaling constant c
    :param sub_sample_frac: float, fraction of data used for estimating the covariance.
        Only takes effect when cov_option=2
    :param groundtruth: array of shape=(n_links, n_features), true value of {\beta^*_i}_{i\in[k]}
    :param record_all_loss:
    """
    def __init__(self, cov_option=1, n_iters=100, tol=1e-3,
                 sub_sample_frac=0.2,
                 groundtruth=None, record_all_loss=False):
        self.cov_option_ = cov_option
        if cov_option not in [1, 2]:
            raise ValueError("Parameter `cov_option` must be either 1 or 2.")
        self.beta_nlr_ = None
        self.eta_ = tol
        self.n_iters_ = n_iters
        assert 0 < sub_sample_frac <= 1
        self.sub_sample_frac_ = sub_sample_frac
        self.record_loss_ = record_all_loss
        self.groundtruth_ = groundtruth

    @property
    def estimator(self):
        if self.beta_nlr_ is None:
            raise NotFittedError("Model hasn't been fitted")
        return self.beta_nlr_

    def loss(self, groundtruth, ord=2, relative=False):
        """Compare the estimated beta v.s. groundtruth beta

        :param groundtruth: array of shape=(n_links, n_features), true value of {\beta^*_i}_{i\in[k]}
        :return err: array of shape=(n_links,), l2-error for each beta^*_i
        """
        if self.beta_nlr_ is None:
            raise NotFittedError("Model hasn't been fitted")
        abs_loss = np.linalg.norm(self.beta_nlr_ - groundtruth,
                                  ord=ord, axis=1)
        if relative:
            beta_norm = np.linalg.norm(groundtruth, ord=ord, axis=1)
            rel_loss = abs_loss / beta_norm
            return rel_loss
        else:
            return abs_loss

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
            Sigma_inv = _sub_sampling_cov(X, s=int(n_samples * self.sub_sample_frac_))

        # compute the ordinary least squares estimator beta_ols
        YZ = Z * Y.reshape(n_samples, 1)
        beta_ols = Sigma_inv.dot(X.T.dot(YZ))  # beta_ols.shape = (n_features, n_links)
        Y_ols = X.dot(beta_ols).T  # Y_ols.shape = (n_links, n_samples)

        # create list of function derivatives and 2nd-order derivatives
        VFprime = [f.vgrad for f in F]
        VFpprime = [f.vggrad for f in F]

        def L_closure(j):
            return lambda c: ((c / n_samples) * np.sum(VFprime[j](Y_ols[j] * c)) - 1)
        L = [L_closure(j) for j in range(n_links)]

        def Lprime_closure(j):
            return lambda c: ((1 / n_samples) * np.sum(VFprime[j](Y_ols[j] * c) +
                                                       c * Y_ols[j] * VFpprime[j](Y_ols[j] * c)))
        L_prime = [Lprime_closure(j) for j in range(n_links)]

        # compute the scale constant for beta_nlr via Newton's method
        C = [_finding_root(L[j], fprime=L_prime[j]) for j in range(n_links)]

        # final value of beta_nlr
        self.beta_nlr_ = (beta_ols * C).T  # beta_nlr.shape=(n_links, n_features)

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


def _finding_root(func, x0=None, fprime=None, interv0=None,
                  method=None, max_iter=100):
    """Finding the root for equation func(x)=0

    :param func: function accepting a real and return a real
    :param x0: real number, the initial guess of
    :param fprime: function, derivative of f
    :param interv0: initial guess of the interval where the root resides
    :param method: str in {'newton', 'brentq'}, default None.
        If None, will first try Newton-Raphson then Brent.
    :param max_iter: int, maximum number of iterations allowed. Default 100.
    :return root: real number, the found root
    """
    # decide the search range for x
    if interv0 is not None:
        left, right = interv0
        assert left > -np.inf and right < np.inf
        assert left < right
    else:
        left, right = -0.05, 0.05
        if func(0) < 0:
            # This is specific for the SLSE task we deal with, where the
            # function is always -1 at x=0. See Algorithm 1 in the paper.
            while True:
                if func(left) > 0:
                    right = 0
                    break
                if func(right) > 0:
                    left = 0
                    break
                if right > 2e12:
                    warnings.warn("The function may not have a root.",
                                  category=UserWarning)
                    break
                left *= 2
                right *= 2
        else:
            left, right = -4096, 4096

    # Use Newton-Raphson Method when derivative is available
    if not (method == 'brentq' or fprime is None):
        interv_len = min(1, (right - left) / 10)
        mid = (right + left) / 2

        if x0 is None:
            x0 = np.random.uniform(low=mid - interv_len,
                                   high=mid + interv_len)

        # find a initial point that's not zero
        for _ in range(11):
            if np.abs(fprime(x0)) > 1e-3:
                break
            interv_len *= 2
            interv_len = min((right - left) / 2, interv_len)
            x0 = np.random.uniform(low=mid - interv_len,
                                   high=mid + interv_len)

        # run Newton-Raphson
        root = newton(func, x0=x0, fprime=fprime,
                      maxiter=max_iter)

        # return if successfully find the root
        if np.abs(func(root)) < 1e-3:
            return root
        elif method == 'newton':
            warnings.warn("Newton-Raphson failed to find the root. Should try other method.\n",
                          category=UserWarning)
            return root

    # Use Brent's Method to find the root
    try:
        r = brentq(func, a=left, b=right, xtol=2e-6, rtol=2e-8,
                   maxiter=max_iter, full_output=False, disp=False)
        return r
    except ValueError:
        # If func(left) * func(right) > 0, Brent's Method will fail
        warnings.warn("Brent's method failed to find the root. "
                      "Fallback to gridsearch", category=UserWarning)
        grid = np.arange(-10, 10, 0.05)
        results = np.vectorize(func)(grid)
        r = results[np.argmin(np.abs(results))]
        return r

