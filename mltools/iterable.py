from itertools import chain


def flatten(iterable):
    return list(chain.from_iterable(iterable))


def argmax(iterable):
    array = list(iterable)
    return max(range(len(array)), key=lambda x: array[x])


def transpose(iterable):
    # iterable should be 1D or 2D array
    return list(zip(*iterable))


def all_equal(iterable):
    array = list(iterable)
    val = array[0]
    return all(x == val for x in array)


def unique(iterable, id_func=None):
    if not id_func:
        def id_func(x): return x
    seen = {}
    result = []
    for item in iterable:
        marker = id_func(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result
