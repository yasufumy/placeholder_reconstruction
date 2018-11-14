from collections import Counter


def build_vocabulary(word_iterator, size=None,
                     special_words=('<unk>', '<s>', '</s>')):
    counter = Counter(word_iterator)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = list(zip(*count_pairs))
    if special_words:
        words = special_words + words
    vocab_size = len(words)
    assigned_size = size if size else vocab_size
    size = assigned_size if assigned_size < vocab_size else vocab_size
    indices = range(size)
    word_to_id = dict(zip(words, indices))

    return word_to_id, words[:size]
