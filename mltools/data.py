import json
from collections import UserList


class Field:
    @staticmethod
    def numericalize(example):
        return example

    @staticmethod
    def preprocess(example):
        return example


class Loader:
    @staticmethod
    def from_json(path, fields):
        with open(path) as json_file:
            items = json.load(json_file)
        examples = {key: Example(field=field) for key, field in fields.items()}
        [examples[key].append(it[key]) for it in items for key, field in fields.items()]
        return examples


class Example(UserList):
    def __init__(self, data=None, field=Field()):
        self.field = field
        super().__init__(data)

    def __getitem__(self, i):
        return self.field.numericalize(self.data[i])

    def __iter__(self):
        for item in self.data:
            yield item

    def append(self, value):
        super().append(self.field.preprocess(value))


class Dataset:
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = fields

    def __getattr__(self, attr):
        if attr in self.examples:
            return self.examples[attr]
        else:
            return getattr(self, attr)
