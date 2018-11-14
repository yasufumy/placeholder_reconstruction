import pytest

from sampling import OrderProvider, Sampling


class TestOrderIterator:

    @pytest.mark.parametrize(('test_input',), [
        ([0, 0, 1, 1, 2, 2, 3, 3],)
    ])
    def test_order_iterator(self, test_input):
        order_provider = OrderProvider(Sampling.get_random_order(len(test_input)))
        order1 = next(order_provider)
        order2 = next(order_provider)
        assert all(a == b for a, b in zip(order1, order2))
        order_provider.update()
        order3 = next(order_provider)
        assert any(a != b for a, b in zip(order1, order3))
