import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from utils.link_funcs import PolynLink, LogisticLink, LogexpLink


class TestUtils(unittest.TestCase):

    def test_PolynLink(self):
        f = PolynLink(degree=3, coeff=2)
        assert f.eval(2) == 16
        assert f.grad(2) == 24
        assert f.ggrad(2) == 24

        f = PolynLink(degree=1, coeff=2)
        assert f.ggrad(2) == 0

        f = PolynLink(degree=0, coeff=2)
        assert f.grad(2) == 0
        assert f.ggrad(2) == 0

    def test_LogisticLink(self):
        f = LogisticLink()
        assert_almost_equal(f.eval(2), 1 / (1 + np.exp(-2)))
        assert_almost_equal(f.grad(2), np.exp(2) / (1 + np.exp(2)) ** 2)

    def test_LogexpLink(self):
        f1 = LogisticLink()
        f2 = LogexpLink()

        assert_almost_equal(f2.grad(2), -f1.eval(-2))
        assert_almost_equal(f2.ggrad(2), f1.grad(2))
