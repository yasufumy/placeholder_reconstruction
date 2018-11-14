import pytest

import iterable

class TestIterable:

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([[1], [2], [3]], [1, 2, 3]),
        ([[[1, 2, 3]]], [[1, 2, 3]])
    ])
    def test_flatten(self, test_input, expected):
        result = iterable.flatten(test_input)
        assert result == expected

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([1, 2, 3, 4, 5], 4)
    ])
    def test_argmax(self, test_input, expected):
        result = iterable.argmax(test_input)
        assert result == expected

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([[1, 2, 3], [4, 5, 6]], [(1, 4), (2, 5), (3, 6)])
    ])
    def test_transpose(self, test_input, expected):
        result = iterable.transpose(test_input)
        assert result == expected

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([1, 1, 1, 1], True),
        ('abcedfg', False)
    ])
    def test_all_equal(self, test_input, expected):
        result = iterable.all_equal(test_input)
        assert result == expected

    @pytest.mark.parametrize(('test_input', 'id_func', 'expected'), [
        ([1, 2, 1, 2, 3, 4], None, [1, 2, 3, 4]),
        ([[1], [1], [2], [1]], tuple, [[1], [2]])
    ])
    def test_unique(self, test_input, id_func, expected):
        result = iterable.unique(test_input, id_func)
        assert result == expected
