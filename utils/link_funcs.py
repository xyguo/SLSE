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

    def grad(self, x):
        """compute the first-order derivative"""
        pass

    def ggrad(self, x):
        """compute the second-order derivative"""
        pass


class PolynLink(LinkFunc):
    """Polynomial link function f(x)=c * x^d"""

    def __init__(self, degree, coeff=1):
        assert degree >= 0
        self.degree_ = degree
        self.coeff_ = coeff

    def eval(self, x):
        return self.coeff_ * np.power(x, self.degree_)

    def grad(self, x):
        if self.degree_ == 0:
            return 0
        return self.degree_ * self.coeff_ * \
               np.power(x, self.degree_ - 1)

    def ggrad(self, x):
        if self.degree_ <= 1:
            return 0
        return self.degree_ * (self.degree_ - 1) * \
               self.coeff_ * np.power(x, self.degree_ - 2)


class LogisticLink(LinkFunc):
    """Logistic link function f(x)=1 / (1 + exp(-x))"""

    def eval(self, x):
        return expit(x)

    def grad(self, x):
        fx = expit(x)
        return fx * (1 - fx)

    def ggrad(self, x):
        fx = expit(x)
        return (fx ** 2) * (2 * fx - 3) + fx


class LogexpLink(LinkFunc):
    """Log-exp link function f(x) = log(1+exp(-x))"""

    def eval(self, x):
        return log1p(np.exp(-x))

    def grad(self, x):
        return -expit(-x)

    def ggrad(self, x):
        g = expit(x)
        return g * (1 - g)


