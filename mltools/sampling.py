import random
from itertools import chain

import numpy as np


class Sampling:
    @classmethod
    def get_uniformly_sampled_order(cls, labels, categories, batch_size):
        if batch_size < categories:
            yield from cls.provide_random_order(len(labels))
        indices = [[k for k, v in enumerate(labels) if v == i]
                   for i in range(categories)]
        sample_size = batch_size // categories
        max_len = max(len(i) for i in indices)

        while True:
            [random.shuffle(index) for index in indices]
            order = list(chain(*(index[pos:pos + sample_size]
                                 for pos in range(0, max_len, sample_size)
                                 for index in indices)))
            yield order

    @staticmethod
    def get_random_order(data_size):
        while True:
            yield np.random.permutation(data_size)

    @staticmethod
    def get_sentence_size_sorted_order(dataset, batch_size):
        order, _ = zip(*sorted(enumerate(dataset), key=lambda x: -len(x[1])))
        chunked_order = [order[pos:pos + batch_size]
                         for pos in range(0, len(order), batch_size)]
        while True:
            random.shuffle(chunked_order)
            yield list(chain.from_iterable(chunked_order))


class OrderProvider:
    def __init__(self, sampler):
        self.current_order = next(sampler)
        self.sampler = sampler

    def __next__(self):
        return self.current_order

    def update(self):
        self.current_order = next(self.sampler)
