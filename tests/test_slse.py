import unittest
import numpy as np
import warnings
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from SLSE.slse import (
    SLSE, _sub_sampling_cov, _finding_root
)
from utils.link_funcs import PolynLink, LogisticLink, LogexpLink
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

        # test if the function falls back to brute force when there's no root
        bad_func = lambda x: -(x ** 2 + 2 * x + 2)
        bad_fprime = lambda x: -(2 * x + 2)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Call some code that triggers a custom warning.
            r = _finding_root(bad_func, x0=-2, fprime=bad_fprime)

            # ignore any non-custom warnings that may be in the list
            w = list(filter(lambda i: issubclass(i.category, UserWarning), w))

            assert len(w) == 2
            assert 'gridsearch' in str(w[-1].message)

    def test_SLSE(self):
        # recall
        # y = \sum^k_{i=1} z_i f_i(\langle \beta^*_i, x \rangle) + \epsilon
        n_samples, n_features = 200000, 5
        n_links = 3
        true_beta = np.random.normal(1, 4, size=(n_links, n_features))
        X = np.random.uniform(low=-1/n_features, high=1/n_features,
                              size=(n_samples, n_features))
        Z = np.random.randn(n_samples, n_links)

        model = SLSE()

        # test polynomial link func
        F1 = [PolynLink(degree=1) for i in range(n_links)]
        Y = gen_slcnr_obj(X, true_beta, F1, Z, eps_std=0.1)
        model.fit(X, Y, Z, F1)
        rel_err = model.loss(true_beta) / np.linalg.norm(true_beta, axis=1)
        assert np.all(rel_err < 0.2)

        rel_err1 = model.loss(true_beta, relative=True)
        assert_array_almost_equal(rel_err, rel_err1)

        # test logistic link func
        F2 = [LogisticLink() for i in range(n_links)]
        Y = gen_slcnr_obj(X, true_beta, F2, Z, eps_std=0.1)
        model.fit(X, Y, Z, F2)
        rel_err = model.loss(true_beta) / np.linalg.norm(true_beta, axis=1)
        assert np.all(rel_err < 0.2)

        rel_err1 = model.loss(true_beta, relative=True)
        assert_array_almost_equal(rel_err, rel_err1)

        # test logexp link func
        F3 = [LogexpLink() for i in range(n_links)]
        Y = gen_slcnr_obj(X, true_beta, F3, Z, eps_std=0.1)
        model.fit(X, Y, Z, F3)
        rel_err = model.loss(true_beta) / np.linalg.norm(true_beta, axis=1)
        assert np.all(rel_err < 0.2)

        rel_err1 = model.loss(true_beta, relative=True)
        assert_array_almost_equal(rel_err, rel_err1)

        # test mixed link func
        F4 = [PolynLink(degree=3), LogexpLink(), LogisticLink()]
        Y = gen_slcnr_obj(X, true_beta, F4, Z, eps_std=0.01)
        model.fit(X, Y, Z, F4)
        rel_err = model.loss(true_beta) / np.linalg.norm(true_beta, axis=1)
        assert np.all(rel_err < 0.2)

        rel_err1 = model.loss(true_beta, relative=True)
        assert_array_almost_equal(rel_err, rel_err1)

