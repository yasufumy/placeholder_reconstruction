import unittest
from unittest.mock import patch

from data import Field, Loader, Example, Dataset


class TestField(unittest.TestCase):
    def setUp(self):
        self.field = Field()

    def test_numericalize_and_preprocess(self):
        example = 1
        result = self.field.numericalize(example)
        self.assertEqual(example, result)
        result = self.field.preprocess(example)
        self.assertEqual(example, result)


class TestLoader(unittest.TestCase):
    def test_from_json(self):
        json_obj = [{'source': [1, 2, 3], 'target': [4, 5, 6]},
                    {'source': [1, 2, 3], 'target': [4, 5, 6]}]
        file_path = '/path/to/json_file'
        import json  # noqa: F401
        open_patcher = patch('data.open')
        json_load_patcher = patch('json.load', return_value=json_obj)
        open_mock = open_patcher.start()
        json_load_mock = json_load_patcher.start()

        source = Field()
        target = Field()
        examples = Loader.from_json(file_path, {'source': source, 'target': target})

        open_mock.assert_called_once_with(file_path)
        json_load_mock.assert_called_once_with(open_mock(file_path).__enter__())

        self.assertEqual(len(examples['source']), 2)
        self.assertEqual(len(examples['target']), 2)
        self.assertListEqual(examples['source'].data, [[1, 2, 3], [1, 2, 3]])
        self.assertListEqual(examples['target'].data, [[4, 5, 6], [4, 5, 6]])

        open_patcher.stop()
        json_load_patcher.stop()


class TestExample(unittest.TestCase):
    def setUp(self):
        self.example = Example()

    def test_init(self):
        self.assertIsInstance(self.example.field, Field)

    @patch.object(Field, 'preprocess')
    def test_append(self, mock):
        item = 1
        mock.return_value = item
        self.example.append(item)
        self.assertTrue(self.example.data[0], 1)
        mock.assert_called_once_with(item)

    @patch.object(Field, 'numericalize')
    def test_getitem(self, mock):
        item = 1

        self.example.append(item)
        self.assertTrue(self.example.data[0], 1)

        mock.return_value = item
        result = self.example[0]
        self.assertEqual(result, item)
        mock.assert_called_with(self.example.data[0])

        mock.return_value = [item]
        result = self.example[0:]
        self.assertListEqual(result, [item])
        mock.assert_called_with(self.example.data[0:])

    def test_iter(self):
        [self.example.append(i) for i in range(10)]
        for i, item in enumerate(self.example):
            self.assertEqual(i, item)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.fields = {'source': Field(), 'target': Field()}
        self.examples = {'source': [1, 2, 3], 'target': [1, 2, 3]}
        self.dataset = Dataset(self.examples, self.fields)

    def test_init(self):
        self.assertEqual(self.dataset.fields, self.fields)
        self.assertEqual(self.dataset.examples, self.examples)

    def test_getattr(self):
        self.assertEqual(self.dataset.source, self.examples['source'])
        self.assertEqual(self.dataset.target, self.examples['target'])


if __name__ == '__main__':
    unittest.main()
