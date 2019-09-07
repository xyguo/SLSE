# -*- coding: utf-8 -*-
"""
Wrapper for some link function definitions used in the experiment

Code Author: Xiangyu Guo
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import expit, log1p


class LinkFunc(object, metaclass=ABCMeta):
    """Base class for link functions"""
    @abstractmethod
    def eval(self, x):
        pass

    def veval(self, X):
        """evaluate the function on all elements in X"""
        return np.apply_along_axis(self.eval, axis=1, arr=X)

    def grad(self, x):
        """compute the first-order derivative"""
        pass

    def vgrad(self, X):
        """compute the first-order derivative for all elements in X"""
        return np.apply_along_axis(self.grad, axis=1, arr=X)

    def ggrad(self, x):
        """compute the second-order derivative"""
        pass

    def vggrad(self, X):
        """compute the second-order derivative for all elements in X"""
        return np.apply_along_axis(self.ggrad, axis=1, arr=X)


class PolynLink(LinkFunc):
    """Polynomial link function f(x)=c * x^d"""

    def __init__(self, degree, coeff=1):
        assert degree >= 0
        self.degree_ = degree
        self.coeff_ = coeff

    def eval(self, x):
        """x is a real number"""
        return self.coeff_ * np.power(x, self.degree_)

    def veval(self, X):
        """X.shape=(n_samples,)"""
        return self.coeff_ * np.power(X, self.degree_)

    def grad(self, x):
        if self.degree_ == 0:
            return 0
        return self.degree_ * self.coeff_ * \
               np.power(x, self.degree_ - 1)

    def vgrad(self, X):
        """compute the first-order derivative for all elements in X"""
        if self.degree_ == 0:
            return 0
        return self.degree_ * self.coeff_ * \
               np.power(X, self.degree_ - 1)

    def ggrad(self, x):
        if self.degree_ <= 1:
            return 0
        return self.degree_ * (self.degree_ - 1) * \
               self.coeff_ * np.power(x, self.degree_ - 2)

    def vggrad(self, X):
        """compute the second-order derivative for all elements in X"""
        if self.degree_ <= 1:
            return 0
        return self.degree_ * (self.degree_ - 1) * \
               self.coeff_ * np.power(X, self.degree_ - 2)


class LogisticLink(LinkFunc):
    """Logistic link function f(x)=1 / (1 + exp(-x))"""

    def eval(self, x):
        return expit(x)

    def veval(self, X):
        return expit(X)

    def grad(self, x):
        fx = expit(x)
        return fx * (1 - fx)

    def vgrad(self, X):
        """compute the first-order derivative for all elements in X"""
        fx = expit(X)
        return fx * (1 - fx)

    def ggrad(self, x):
        fx = expit(x)
        return fx * (1 - fx) * (1 - 2 * fx)

    def vggrad(self, X):
        """compute the second-order derivative for all elements in X"""
        fx = expit(X)
        return fx * (1 - fx) * (1 - 2 * fx)


class LogexpLink(LinkFunc):
    """Log-exp link function f(x) = log(1+exp(-x))"""

    def eval(self, x):
        return log1p(np.exp(-x))

    def veval(self, X):
        return log1p(np.exp(-X))

    def grad(self, x):
        return -expit(-x)

    def vgrad(self, X):
        """compute the first-order derivative for all elements in X"""
        return -expit(-X)

    def ggrad(self, x):
        g = expit(x)
        return g * (1 - g)

    def vggrad(self, X):
        """compute the second-order derivative for all elements in X"""
        g = expit(X)
        return g * (1 - g)


