from itertools import chain
from operator import itemgetter
from math import log, exp
import pickle
from argparse import ArgumentParser

import numpy
import autograd.numpy as np
from autograd import grad
from nltk.tokenize import word_tokenize
from sklearn.utils.extmath import logsumexp

from mltools.vocabulary import build_vocabulary
from mltools.data import Field

from dataset import OptaDataset, EventField


class CommentField(Field):
    @staticmethod
    def preprocess(example):
        return example['id']


class DictEventField(EventField):
    @staticmethod
    def preprocess(example):
        return {x['id']: [x['type_id'], x['player_id'], x['team_id'], x['outcome'], x['x'],
                x['y'], x['end_x'], x['end_y'], [tuple(i) for i in x['details']]]
                for x in example}

    def build_vocabulary(self, examples, size=None, special_words=None):
        examples = [ex.values() for ex in examples]
        super().build_vocabulary(examples, size, special_words)

    def numericalize(self, example):
        keys = example.keys()
        tensor = super().numericalize([example[k] for k in keys])
        return {k: v for k, v in zip(keys, tensor)}


def preprocessor(string):
    return word_tokenize(str.lower(string))


class AlignmentDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        with open(self.data_path) as f:
            dataset = []
            for line in f:
                comment, event_type, game_id, comment_id = line.strip().split('\t')
                comment = preprocessor(comment)
                event_types, event_ids = zip(*[x.split(':') for x in event_type.split(' ')])
                event_dict = {}
                for i, t in zip(event_ids, event_types):
                    if t not in event_dict:
                        event_dict[t] = []
                    event_dict[t].append(i)
                dataset.append((comment, event_types, event_dict, game_id, comment_id))
        comments, event_types_list, event_dict_list, game_ids, comment_ids = zip(*dataset)
        comment_ids = {comment_id: i for i, comment_id in enumerate(comment_ids)}
        return comments, event_types_list, event_dict_list, game_ids, comment_ids


class EMAlgorism:
    def __init__(self, comments, event_types_list, max_iteration, comment_ids, event_dict_list):
        # P(C|D) C:event_type, D:document
        p_cd = [{t: 1. / len(types) for t in types} for types in event_types_list]
        # P(W|C) W:word, C:event_type
        p_wc = {}
        # N(C) C:event_type, counts of event_type
        N_c = {}
        for p_cd_i, comment in zip(p_cd, comments):
            for t, _ in p_cd_i.items():
                if t not in p_wc:
                    p_wc[t] = {}
                N_c[t] = N_c.get(t, 0) + len(comment)
                for word in comment:
                    p_wc[t][word] = 0.
        self.p_cd = p_cd
        self.p_wc = p_wc
        self.N_c = N_c
        self.comments = comments
        self.max_iteration = max_iteration
        self.eps = 1e-6
        self.comment_ids = comment_ids
        self.event_dict_list = event_dict_list

    def _e_step(self):
        # update P(Ci|Di) = multi_j(P(Wj|Ci))
        eps = self.eps
        p_wc = self.p_wc
        for i, (p_cd_i, comment) in enumerate(zip(self.p_cd, self.comments)):
            Z = []  # for normalization
            for event_type in p_cd_i.keys():
                log_p_cd_i = sum([log(p_wc[event_type][word] + eps) for word in comment], 0.)
                self.p_cd[i][event_type] = log_p_cd_i
                Z.append(log_p_cd_i)
            down = logsumexp(numpy.asarray(Z))
            for event_type, log_p_cd_i in self.p_cd[i].items():
                self.p_cd[i][event_type] = exp(log_p_cd_i - down)

    def _m_step(self):
        for i, (p_cd_i, comment) in enumerate(zip(self.p_cd, self.comments)):
            for event_type, p_cd_ij in p_cd_i.items():
                for word in comment:
                    self.p_wc[event_type][word] += p_cd_ij
                for word in comment:
                    self.p_wc[event_type][word] /= self.N_c[event_type]

    def run(self):
        for _ in range(self.max_iteration):
            self._e_step()
            self._m_step()

    def compute_events(self, comment_id):
        i = self.comment_ids.get(comment_id, None)
        if i is None:
            return None
        classes, _ = zip(*sorted(self.p_cd[i].items(), key=itemgetter(1)))
        event_dict = self.event_dict_list[i]
        events = []
        for c in classes:
            events.extend(event_dict[c])
        return events

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def softmax(x):
    y = np.atleast_2d(x)
    axis = 1
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(x.shape) == 1:
        p = p.flatten()
    return p


class FeaturedEM:
    def __init__(self, comments, event_features, max_iteration, learning_rate,
                 m_step_iteration=500):
        self.comments = comments
        self.event_features = event_features
        self.p_cd = [{event_id: 1. / len(xs) for event_id in xs}
                     for xs in event_features]
        self.p_dc = [{event_id: 1. / len(xs) for event_id in xs}
                     for xs in event_features]
        feature_length = max(len(x) for xs in event_features for x in xs.values())
        self.weight = np.random.rand(feature_length)
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.m_step_iteration = m_step_iteration
        self.grad_q_func = grad(self.q_func)

    def q_func(self, w, xs, q):
        y = np.array([w @ x for x in xs])
        p = softmax(y)
        return np.sum(q * np.log(p + 1e-6))

    def _e_step(self):
        for i, p_dc in enumerate(self.p_dc):
            down = sum(p_dc.values())
            for t, p_dc_i in p_dc.items():
                self.p_cd[i][t] = p_dc_i / down

    def _m_step(self):
        w = self.weight
        lr = self.learning_rate
        data_length = len(self.event_features)
        for _ in range(self.m_step_iteration):
            grads = 0.
            for i, (p_cd, xs) in enumerate(zip(self.p_cd, self.event_features)):
                types = p_cd.keys()
                q = np.array([p_cd[t] for t in types])
                xs = np.array([xs[t] for t in types])
                grads += self.grad_q_func(w, xs, q)
            w += grads / data_length * lr
        self.weight = w
        for i, xs in enumerate(self.event_features):
            for t, x in xs.items():
                self.p_dc[i][t] = w @ x

    def run(self):
        for i in range(self.max_iteration):
            print('Iteration {}'.format(i))
            print('e step...')
            self._e_step()
            print('m step...')
            self._m_step()

    def compute_events(self, comment_id):
        i = self.comments.get(comment_id, None)
        if i is None:
            return None
        events, _ = zip(*sorted(self.p_cd[i].items(), key=itemgetter(1)))
        return events

    def save(self, filename):
        del self.grad_q_func
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./result/em.pickle')
    parser.add_argument('--data-path', type=str, default='./dataset/align.txt')
    parser.add_argument('--model-type', type=str, default='em',
                        choices=('em', 'feature-em'))
    parser.add_argument('--iter', type=int, default=10)
    return parser.parse_args()


def main(args):
    if args.model_type == 'em':
        comments, event_types_list, event_dict_list, game_ids, comment_ids = \
            AlignmentDataset(args.data_path).load()
        word_to_id, id_to_word = build_vocabulary(chain.from_iterable(comments), special_words=None)
        em = EMAlgorism(comments, event_types_list, args.iter, comment_ids, event_dict_list)
        em.run()
        em.save(args.model_path)
    elif args.model_type == 'feature-em':
        source = DictEventField()
        target = CommentField()
        data = OptaDataset(path=args.data_path,
                           fields={'source': source, 'target': target})
        source.build_vocabulary(data.source)
        comments = {c: i for i, c in enumerate(data.target)}
        event_features = [source.numericalize(x) for x in data.source]
        em = FeaturedEM(comments, event_features, args.iter, 0.01)
        em.run()
        em.save(args.model_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
