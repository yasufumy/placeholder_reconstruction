import pytest
import numpy as np

import feature

class TestFeature:

    @pytest.mark.parametrize(('test_input', 'expected'), [
        (['i', 'am', 'ok'],
         np.array([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])),
    ])
    def test_co_occurr(self, test_input, expected):
        mat = feature.CoOccurrenceMatrix()
        result = mat.get(test_input)
        assert np.all(result == expected)

    @pytest.mark.parametrize(('test_input', 'expected'), [
        (['i', 'am', 'ok'],
         0),
    ])
    def test_pmi(self, test_input, expected):
        pmi = feature.PMI(test_input)
        result = pmi.calc('i', 'ok')
        assert result == expected

