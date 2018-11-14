import unittest
import pytest

import preprocessing


class TestTextPreprocessing:
    @pytest.mark.parametrize(('test_input', 'expected'),
                             [('abcdefg©\n', [1, 2, 3, 4, 5, 6, 7, 0, 70])])
    def test_String2Tensor(self, test_input, expected):
        s2t = preprocessing.String2Tensor(preprocessing.char_table, 0)
        result = s2t.encode(test_input)
        assert result == expected
        limit = 3
        s2t = preprocessing.String2Tensor(preprocessing.char_table, 0, limit=limit)
        result = s2t.encode(test_input)
        assert result == expected[:limit]
        s2t = preprocessing.String2Tensor(preprocessing.char_table, 0, 10, 11,
                                          limit=3)
        result = s2t.encode(test_input)
        assert result == [10, 1, 11]

    @pytest.mark.parametrize(('test_input', 'expected'),
                             [('I am a student. Great!!!!', 'I am a student Great')])
    def test_remove_ignore_chars(self, test_input, expected):
        result = preprocessing.remove_ignore_chars(test_input)
        assert result == expected


class TestPreprocessing(unittest.TestCase):
    def test_pad(self):
        test_input = [[1, 2, 3], [4, 5]]
        expected = [[1, 2, 3], [4, 5, -1]]
        pad = preprocessing.Pad(3, -1)
        result = pad(test_input)
        self.assertListEqual(result, expected)
