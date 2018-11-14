import unittest

import numpy as np
from numpy.testing import assert_array_equal

from utils import compute_class_weight


class TestUtils(unittest.TestCase):
    def test_compute_class_weight(self):
        players = ['john', 'jack']
        word_to_id = {'john': 0, 'jack': 1, 'the': 2}
        result = compute_class_weight(players, word_to_id)
        expected = np.array([1.2, 1.2, 1.], dtype=np.float32)
        assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
