import pytest
import numpy as np

import iterator


class TestIterator:

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([0, 1, 2, 3, 4, 5, 6], (0, 1))
    ])
    def test_Iterator(self, test_input, expected):
        test_iter = iterator.Iterator(test_input, 2, gpu=None)
        test_iter = iter(test_iter)
        result = next(test_iter)
        assert result == expected
        test_iter1 = iterator.Iterator(test_input, 2, gpu=None)
        test_iter2 = iterator.Iterator(test_input, 2, gpu=None)
        for t1, t2 in zip(test_iter1, test_iter2):
            assert t1 == t2
        test_iter = iterator.Iterator(test_input, 1, gpu=None)
        test_iter = iter(test_iter)
        result = next(test_iter)
        assert len(result) == 1

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([[[1, 2, 3]], [[4, 5, 6]]], [[[[1, 2, 3]]], [[[4, 5, 6]]]])
    ])
    def test_ImageIterator(self, test_input, expected):
        test_iter = iterator.ImageIterator(test_input, 2, gpu=None)
        test_iter = iter(test_iter)
        result = next(test_iter).data
        assert np.all(result == expected)
