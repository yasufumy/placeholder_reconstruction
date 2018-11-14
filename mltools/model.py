import os

import chainer
from chainer import cuda
from chainer import serializers


class BaseModel(chainer.Chain):

    model_index = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def use_gpu(self, gpu_id):
        cuda.get_device_from_id(gpu_id).use()
        self.to_gpu()

    def save_model(self, filename, save_format='hdf5', suffix=True):
        if suffix:
            root, ext = os.path.splitext(filename)
            filename = '{}{}{}'.format(root, self.model_index, ext)
            self.model_index += 1
        gpu_flag = False
        if not self._cpu:
            self.to_cpu()
            gpu_flag = True
        getattr(serializers, 'save_{}'.format(save_format))(filename, self)
        if gpu_flag:
            self.to_gpu()

    def load_model(self, filename, load_format='hdf5'):
        getattr(serializers, 'load_{}'.format(load_format))(filename, self)
