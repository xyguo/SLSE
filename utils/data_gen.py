# -*- coding: utf-8 -*-
"""Helper functions for creating synthesized data"""
import numpy as np


def gen_slcnr_obj(X, beta, F, Z=None, z_std=1.0, eps=None, eps_std=1.0):
    """Generate sythesized objective value according to the Stochastic
    Linear Combination of Non-linear Regression (SLCNR) model
    y = \sum^k_{i=1} z_i f_i(\langle \beta_i, x \rangle) + \epsilon
    :param X: array of shape=(n_samples, n_features), feature vector
    :param beta: array of shape=(n_links, n_features), weight vector
    :param F: list of len=n_links, of which each element f is an object offering the two following methods:
        1. f.eval(x) -> real, which evaluate f at real number x
        2. f.grad(x) -> real, which evaluate f's derivative at real number x
        3. f.ggrad(x) -> real, which evaluate f's 2nd-order derivative at real number x
    :param Z: array of shape=(n_samples, n_links), random coefficient for combining the link functions.
        If None, then will be generated via Uniform(-z_var, z_var)
    :param z_std: float, standard variance used for generating z
    :param eps: array of shape=(n_samples,), random perturbation of objective value.
        If None, then will be generated vie Gaussian(0, 0.5)
    :param eps_std: float, standard variance used for generating eps
    """
    n_samples, n_features = X.shape
    n_links = len(F)
    if Z is None:
        Z = np.random.uniform(-z_std, z_std, size=(n_samples, n_links))
    if eps is None:
        eps = np.random.normal(0, scale=eps_std, size=(n_samples,))

    Xbeta = X.dot(beta.T)  # Xbeta.shape = (n_samples, n_links)
    VF = [f.veval for f in F]
    fXbeta = [VF[i](Xbeta[:, i]) for i in range(n_links)]
    fXbeta = np.vstack(fXbeta)  # fXbeta.shape = (n_links, n_samples)
    y = np.sum(Z.T * fXbeta, axis=0) + eps
    return y
