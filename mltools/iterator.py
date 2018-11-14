import math

import numpy as np
import cupy as cp
import chainer
import chainer.functions as F

from mltools.preprocessing import Pad


class Iterator:
    def __init__(self, dataset, batch_size, order_provider=None,
                 wrapper=None, gpu=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.wrapper = wrapper or self._wrapper
        data_length = len(dataset)
        if order_provider is not None:
            self.order_provider = order_provider
        else:
            self.order_provider = self._provide_order(data_length)
        self.data_length = data_length
        if gpu is not None:
            self.xp = cp
            cp.cuda.Device(gpu).use()
        else:
            self.xp = np
        self.reset()

    def get_batch(self, order):
        return tuple(self.dataset[i] for i in order)

    @staticmethod
    def _wrapper(batch):
        return batch

    @staticmethod
    def _provide_order(data_length):
        while True:
            yield range(data_length)

    def _provide_batches(self):
        batch_size = self.batch_size
        for i in range(0, self.data_length, batch_size):
            batch = self.get_batch(self.order[i:i + batch_size])
            yield self.wrapper(batch)

    def reset(self):
        self.order = next(self.order_provider)
        self.batches = self._provide_batches()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        for batch in self.batches:
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return math.ceil(self.data_length / self.batch_size)


class SequentialIterator(Iterator):
    def __init__(self, dataset, batch_size, order_provider=None,
                 fix_length=None, fillvalue=-1, gpu=None, dtype='int32'):
        super().__init__(dataset, batch_size, order_provider, None, gpu)
        self.pad = Pad(fix_length, fillvalue)
        self.dtype = getattr(self.xp, dtype)

    def _wrapper(self, batch):
        xp = self.xp
        return F.transpose_sequence(xp.asarray(self.pad(batch), dtype=self.dtype))


class ImageIterator(Iterator):
    def _wrapper(self, batch):
        xp = self.xp
        batch = xp.asarray(batch, dtype=xp.float32)
        batch_size, height, width, *dim = batch.shape
        dim = dim[0] if dim else 1
        return F.reshape(batch, (batch_size, dim, height, width))


class LabelIterator(Iterator):
    def _wrapper(self, batch):
        xp = self.xp
        return chainer.Variable(xp.asarray(batch, dtype=xp.int32))
