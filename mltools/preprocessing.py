char_table = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
              'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
              'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
              'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29,
              '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36,
              '-': 37, ',': 38, ';': 39, '.': 40, '!': 41, '?': 42, ':': 43,
              '\'': 44, '"': 45, '/': 46, '\\': 47, '|': 48, '_': 49, '@': 50,
              '#': 51, '$': 52, '%': 53, '^': 54, '&': 55, '*': 56, '~': 57,
              '`': 58, '+': 59, '=': 60, '<': 61, '>': 62, '(': 63, ')': 64,
              '[': 65, ']': 66, '{': 67, '}': 68, ' ': 69, 'unk': 0, '\n': 70}


class String2Tensor:
    def __init__(self, table, unk_id, start_id=None, end_id=None, limit=None):
        self.table = table
        self.unk_id = unk_id
        self.start = [start_id] if start_id else []
        self.end = [end_id] if end_id else []
        if limit is not None:
            self.limit = limit - len(self.start + self.end)
            self._encode = self._encode_with_padding
        else:
            self._encode = self._encode_without_padding

    def encode(self, sequence):
        return self._encode(sequence)

    def _encode_without_padding(self, sequence):
        table = self.table
        unk_id = self.unk_id
        return self.start + [table.get(s, unk_id) for s in sequence] + self.end

    def _encode_with_padding(self, sequence):
        return self._encode_without_padding(sequence[:self.limit])


def remove_ignore_chars(text, ignore_chars=',.?!:;'):
    for c in ignore_chars:
        if c in text:
            text = text.replace(c, '')
    return text


class Pad:
    def __init__(self, fix_length, fillvalue=None):
        self.fix_length = fix_length
        self.fillvalue = fillvalue

    def __call__(self, items):
        fix_length = self.fix_length
        fillvalue = self.fillvalue
        return [item + [fillvalue] * (fix_length - len(item))
                for item in items]
