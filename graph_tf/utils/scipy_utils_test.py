import unittest

import numpy as np
import scipy.sparse.linalg as la

from graph_tf.utils import scipy_utils


class ScipyUtilsTest(unittest.TestCase):
    def test_shifted_linear_operator(self, n: int = 100, m: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, n))
        x += x.T

        w, v = la.eigsh(x, k=m * 2)
        shifted = scipy_utils.ShiftedLinearOperator(x, v[:, :m], -w[:m])
        wd, vd = la.eigsh(shifted, k=m)
        v *= np.sign(v[:1])
        vd *= np.sign(vd[:1])
        np.testing.assert_allclose(w[m : 2 * m], wd)
        np.testing.assert_allclose(v[:, m : 2 * m], vd)


if __name__ == "__main__":
    unittest.main()
