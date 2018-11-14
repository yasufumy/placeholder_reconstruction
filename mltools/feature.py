import math
from collections import Counter

import numpy as np

from mltools.vocabulary import build_vocabulary


def calc_so(phrase, pos_seed, neg_seed):
    pass


class PMI:
    def __init__(self, word_tokens, window_size=1):
        cooccurr = CoOccurrenceMatrix(window_size)
        self.mat = cooccurr.get(word_tokens)
        self.word_to_id = cooccurr.word_to_id
        self.vocabulary = cooccurr.vocabulary
        self.N = len(word_tokens)

    def calc(self, word1, word2):
        mat = self.mat
        w2i = self.word_to_id
        one = w2i[word1]
        two = w2i[word2]
        log_in = max(self.N * mat[one, two] / mat[one, one] / mat[two, two], 1)
        return math.log(log_in, 2)


class CoOccurrenceMatrix:
    def __init__(self, window_size=1):
        self.window_size = window_size

    def get(self, word_tokens):
        w2i, words = build_vocabulary(word_tokens, special_words=None)
        vocab_size = len(words)
        window_size = self.window_size
        mat = np.zeros((vocab_size, vocab_size))
        for i, w in enumerate(word_tokens):
            mat[w2i[w], w2i[w]] += 1
            for j in range(max(i - window_size, 0), i):
                mat[w2i[w], w2i[word_tokens[j]]] += 1
            i_start = i + 1
            for j in range(i_start, min(i_start + window_size, vocab_size)):
                mat[w2i[w], w2i[word_tokens[j]]] += 1
        self.word_to_id = w2i
        self.vocabulary = words
        return mat


class Tfidf:
    def __init__(self, documents, word_to_id):
        self.documents = documents
        self.word_to_id = word_to_id
        self.N = len(documents)

    def _tf(self, words_count, vocab_size, word_to_id):
        tf = np.zeros((vocab_size,))
        for word, count in words_count.items():
            if word in word_to_id:
                tf[word_to_id[word]] = count
        return tf

    def _idf(self, word, words_count, N):
        return N / sum((count[word] for count in words_count if word in count), 1)

    def compute(self):
        word_to_id = self.word_to_id
        N = self.N
        vocab_size = len(word_to_id)
        tf_values = np.zeros((N,  vocab_size))
        words_count = [Counter(document) for document in self.documents]
        for i, count in enumerate(words_count):
            tf_values[i] = self._tf(words_count[i], vocab_size, word_to_id)
        idf_values = np.zeros((vocab_size,))
        for word, i in word_to_id.items():
            idf_values[i] = self._idf(word, words_count, N)
        idf_values = np.log10(idf_values)
        tf_values = tf_values / np.expand_dims(tf_values.sum(axis=1), axis=1)
        self.tfidf_values = tf_values * idf_values
        return self.tfidf_values
