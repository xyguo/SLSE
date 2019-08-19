import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from utils.data_gen import gen_slcnr_obj
from utils.link_funcs import PolynLink


class TestUtils(unittest.TestCase):

    def test_gen_slcnr_obj(self):
        F = [PolynLink(degree=1),
             PolynLink(degree=2, coeff=2),
             PolynLink(degree=0)]
        true_beta = np.array(
            [[1, 1],
             [0.5, 2],
             [2, 0.5]]
        )
        X = np.array(
            [[1, 1],
             [-1, -1],
             [-1, 1],
             [1, -1],
             [0, 0]]
        )
        Z = np.array(
            [[0.5, -0.5, 0.05],
             [-0.5, 0.01, 0.05],
             [0.2, 0.05, -0.5],
             [-0.05, 0.5, 0.5],
             [-0.2, -0.05, -0.1]]
        )
        Y_true = np.array([-5.2, 1.175, -0.275, 2.75, -0.1])

        Y1 = gen_slcnr_obj(X, true_beta, F, Z, eps=np.zeros(5))
        Y2 = gen_slcnr_obj(X, true_beta, F, Z, eps_std=0.001)

        assert_array_equal(Y_true, Y1)
        assert_array_almost_equal(Y_true, Y2, decimal=2)

