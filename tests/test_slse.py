import unittest
import numpy as np
import warnings
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from SLSE.slse import (
    SLSE, _sub_sampling_cov, _finding_root
)
from utils.link_funcs import PolynLink
from utils.data_gen import gen_slcnr_obj


class TestUtils(unittest.TestCase):

    def test_sub_sampling_OLS(self):
        X = np.random.randn(20, 3)
        s = 5
        S = _sub_sampling_cov(X, s)
        assert S.shape == (3, 3)

    def test_finding_root(self):
        # test if the function indeed find the root when it really exists
        func = lambda x: x ** 3 - 3 * x + 1
        fprime = lambda x: 3 * x ** 2 - 3
        r = _finding_root(func, x0=-2, fprime=fprime, method='newton')
        assert_almost_equal(r, -1.879385233, decimal=4)

        r = _finding_root(func, x0=-3, method='brentq')
        assert_almost_equal(r, -1.879385233, decimal=4)

        # test if the function fails when there's no root
        bad_func = lambda x: x ** 2 + 2 * x + 2
        bad_fprime = lambda x: 2 * x + 2
        try:
            r = _finding_root(bad_func, x0=-2, fprime=bad_fprime)
        except ValueError:
            pass
        else:
            assert False

    def test_SLSE(self):
        # recall
        # y = \sum^k_{i=1} z_i f_i(\langle \beta^*_i, x \rangle) + \epsilon
        F = [PolynLink(degree=1),
             PolynLink(degree=1, coeff=2),
             PolynLink(degree=3)]
        true_beta = np.array(
            [[1, 1],
             [0.5, 2],
             [2, 0.5]]
        )
        X = np.random.normal(loc=2, scale=5, size=(10, 2))
        Z = np.random.randn(10, 3)
        Y = gen_slcnr_obj(X, true_beta, F, Z, eps_std=0.2)

        model = SLSE()
        model.fit(X, Y, Z, F)
        assert_array_almost_equal(true_beta, model.estimator, decimal=2)
