import os
import random
from glob import glob
import linecache
from argparse import ArgumentParser
from string import Template


class FileTemplate(Template):
    def __init__(self, filename):
        with open(filename) as f:
            text = f.read()
        super().__init__(text)


class LazyLoad:
    def __init__(self, filename, shuffle=True, seed=123):
        self._filename = filename
        self._total_lines = 0
        with open(filename) as f:
            self._total_lines = len(f.readlines()) - 1
        self._indices = list(range(self._total_lines))
        self._seed = seed
        if shuffle:
            self.shuffle()
        self._shuffle = shuffle

    def pick(self):
        if len(self._indices) > 0:
            i = self._indices.pop()
            return linecache.getline(self._filename, i)
        else:
            return ''

    def shuffle(self):
        random.seed(self._seed)
        random.shuffle(self._indices)

    def reset(self):
        self._indices = list(range(self._total_lines))
        self.shuffle()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--template', default='./body_template.txt', type=str)
    parser.add_argument('--ref', default='', type=str)
    parser.add_argument('--source', default='', type=str)
    parser.add_argument('--target', default='', type=str)
    parser.add_argument('--length', default=100, type=int)
    parser.add_argument('--split', default=4, type=int)
    parser.add_argument('--seed', default=12345, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.length % args.split != 0:
        raise ValueError('args.length / args.split should be divisible')
    size = args.length // args.split
    indices = [size * i for i in range(args.split)]
    pathes = glob(os.path.join(args.source, '*.txt'))
    files = [LazyLoad(path, seed=args.seed) for path in pathes]
    models = {'model{}'.format(i): os.path.basename(path).replace('.txt', '')
              for i, path in enumerate(pathes, start=1)}
    ref = LazyLoad(args.ref, seed=args.seed)

    for start in indices:
        body = ''
        end = start + size
        for i, _ in enumerate(range(start, end), start=1):
            template = FileTemplate(args.template)
            hypos = {'hypothesis{}'.format(j): f.pick()
                     for j, f in enumerate(files, start=1)}
            kwargs = {**models, **hypos}
            body += '{}\n'.format(
                template.substitute(num=i, reference=ref.pick(), **kwargs))

        main = FileTemplate('./main_template.txt')

        with open('{}_{}-{}'.format(args.target, start, end), 'w') as f:
            f.write(main.safe_substitute(body=body))
