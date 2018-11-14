import os
from collections.abc import Sequence


class TextFile:
    def __init__(self, filename, data):
        self.filename = filename
        if isinstance(data, Sequence) and not isinstance(data, str):
            self._write = self._write_sequence
        else:
            self._write = self._write_data
        self.data = data

    def save(self):
        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.filename, 'w', encoding='utf-8') as fp:
            self._write(fp, self.data)

    @staticmethod
    def _write_sequence(fp, sequence):
        [fp.write('{}\n'.format(x)) for x in sequence]

    @staticmethod
    def _write_data(fp, data):
        fp.write('{}'.format(data))
