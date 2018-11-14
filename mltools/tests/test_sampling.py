from itertools import chain

import pytest

from sampling import Sampling


class TestSampling:

    @pytest.mark.parametrize(('test_input', 'expected'), [
        ([[0, 0, 1, 1, 2, 2, 3, 3]], [[0, 1, 2, 3, 0, 1, 2, 3]])
    ])
    def test_uniformly_sampling(self, test_input, expected):
        batch_size = 4
        order_provider = Sampling.get_uniformly_sampled_order(
                        test_input, 4, batch_size)
        order = next(order_provider)
        result = chain.from_iterable([[test_input[ind] for ind in order[i:i+batch_size]]
                                     for i in range(0, len(test_input), batch_size)])
        assert all(a == b for a, b in zip(result, expected))

    @pytest.mark.parametrize(('test_input', 'expected'), [
        (4, {0, 1, 2, 3})
    ])
    def test_random_sampling(self, test_input, expected):
        order_provider = Sampling.get_random_order(test_input)
        order = next(order_provider)
        result = set(order)
        assert result == expected

    @pytest.mark.parametrize(('test_input', 'expected'), [
        (['aaa', 'aa', 'a', 'aaaa'], {3, 0, 1, 2})
    ])
    def test_sentence_size_sorted_sampling(self, test_input, expected):
        order_provider = Sampling.get_sentence_size_sorted_order(test_input, 2)
        order = next(order_provider)
        result = set(order)
        assert result == expected
