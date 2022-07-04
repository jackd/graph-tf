import os
import unittest

import numpy as np

from graph_tf.utils import np_utils
from graph_tf.utils.temp_utils import tempfile_context


class NpUtilsTest(unittest.TestCase):
    def test_array_stream_npz(self):

        with tempfile_context() as root:
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, "data.npz")
            x = np.random.uniform(size=(5, 3)).astype(np.float32)
            y = (np.random.uniform(size=(3, 4)) * 10).astype(np.int64)
            np.savez(path, x=x, y=y)

            def assert_same(stream: np_utils.ArrayStream, data: np.ndarray):
                assert len(stream) == data.shape[0]
                assert stream.shape == data.shape
                assert stream.dtype == data.dtype
                for i, xi in enumerate(stream):
                    np.testing.assert_array_equal(xi, data[i])
                assert i == data.shape[0] - 1  # pylint: disable=undefined-loop-variable

            assert_same(np_utils.ArrayStream.from_npz(path, "x"), x)
            assert_same(np_utils.ArrayStream.from_npz(path, "y"), y)

            npz = np.load(path)
            assert_same(np_utils.ArrayStream.from_npz(npz, "x"), x)
            assert_same(np_utils.ArrayStream.from_npz(npz, "y"), y)


if __name__ == "__main__":
    unittest.main()
