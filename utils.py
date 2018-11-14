import os
import re
import json
import copy
from pathlib import Path
from collections.abc import Sequence
from collections import deque
from datetime import datetime
from itertools import takewhile, chain

import chainer
import yaml
import numpy as np
import cupy as cp
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from mltools.trainer import Seq2SeqTrainer, generate_file_writer, print
from functions import binary_each_accuracy


def flatten(seq):
    return list(chain.from_iterable(seq))


class Utility:
    @staticmethod
    def make_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def now_string():
        return datetime.now().strftime('%m%d%y%H%M%S')

    @staticmethod
    def args_to_directory(args, ignore=None):
        args = copy.deepcopy(args)
        args_dict = vars(args)
        if ignore is not None:
            for key in ignore:
                del args_dict[key]
        tmp = ''.join(['{}{}'.format(k, v) for k, v in args_dict.items()])
        return tmp.replace('/', '_')

    @staticmethod
    def get_save_directory(model, save_path):
        fill_width = 5
        directories = sorted(Path(save_path).glob('{}_*'.format(model)))
        indices = [i for i, x in enumerate(directories)
                   if str(i).zfill(fill_width) not in x.name]
        if directories and not indices:
            last_dir = directories[-1].name
            new_dir = re.sub('(\d+)(?!\d)',
                             lambda x: str(int(x.group(0)) + 1).zfill(fill_width),
                             last_dir)
        elif indices:
            new_dir = '{}_{}'.format(model, str(indices[0]).zfill(fill_width))
        else:
            new_dir = '{}_00000'.format(model)
        return new_dir


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


def compute_class_weight(player_file_path, word_to_id, player_weight=1.2,
                         other_weight=1., gpu=None):
    if gpu is not None:
        xp = cp
        cp.cuda.Device(gpu).use()
    else:
        xp = np
    class_weight = xp.full(len(word_to_id), other_weight, dtype=xp.float32)

    with open(player_file_path) as fp:
        players = []
        for line in fp:
            players.extend(line.strip().split())

    player_indices = []
    for p in players:
        if p in word_to_id:
            ind = word_to_id[p]
            if ind not in player_indices:
                player_indices.append(ind)
    class_weight[player_indices] = player_weight
    return class_weight


class EndTokenIdRemoval:
    def __init__(self, end_token_id):
        self.is_end_token_id = lambda x: x != end_token_id

    def __call__(self, batch):
        return [list(takewhile(self.is_end_token_id, x)) for x in batch]


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Seq2SeqWithLabelTrainer(Seq2SeqTrainer):
    def train_one_epoch(self):
        sum_loss = 0
        for x_batch, t_batch in zip(tqdm(self.sources_train, desc='train'),
                                    self.targets_train):
            self.model.cleargrads()
            loss, y_batch = self.model.loss(x_batch, t_batch)
            loss.backward()
            self.optimizer.update()
            sentence_length = len(t_batch[0])
            batch_size = t_batch[0][1].shape[0]
            # returned loss
            # batch_averaged_loss (output of softmax_cross_entropy) * sentence_length
            sum_loss += loss.data * batch_size / sentence_length
            del loss
        self.order_provider.update()
        return sum_loss / self.data_length

    def run(self):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fp_loss = open(os.path.join(save_path, 'loss.txt'), 'w', encoding='utf-8')
        fp_bleu = open(os.path.join(save_path, 'bleu.txt'), 'w', encoding='utf-8')
        writer_loss = generate_file_writer(fp_loss)
        writer_bleu = generate_file_writer(fp_bleu)
        early_stopping_size = max(3, 6 - self.evaluation_step)
        evaluation_scores = deque(maxlen=early_stopping_size)
        best_model = None
        best_score = -1
        # model_name = '{}.model'.format(self.model.name)
        # model_filename = os.path.join(save_path, model_name)
        for i in tqdm(range(self.epoch), desc='epoch'):
            epoch = i + 1
            print('epoch: {}'.format(epoch))
            with chainer.using_config('train', True), \
                    chainer.using_config('enable_backprop', True):
                average_loss = self.train_one_epoch()
            print('loss: {}'.format(average_loss))
            writer_loss(average_loss)
            if epoch % self.evaluation_step == 0:
                with chainer.using_config('train', False), \
                        chainer.using_config('enable_backprop', False):
                    bleu, accuracy, hypotheses = self.evaluater(self.model, self.sources_train2,
                                                                self.targets_train2)
                print('train BLEU score: {}'.format(bleu))
                with chainer.using_config('train', False), \
                        chainer.using_config('enable_backprop', False):
                    bleu, accuracy, hypotheses = self.evaluater(self.model, self.sources_test,
                                                                self.targets_test)
                print('dev BLEU score: {}'.format(bleu))
                print('encoder accuracy: {}'.format(accuracy))
                # self.model.save_model(model_filename)
                evaluation_scores.append(bleu)
                writer_bleu(bleu)
                if best_score < bleu:
                    best_score = bleu
                    best_model = self.model.copy().to_cpu()
            if self.early_stopping and len(evaluation_scores) >= early_stopping_size:
                # empty list is False
                is_improved = [evaluation_scores[i] < evaluation_scores[i + 1]
                               for i in range(len(evaluation_scores) - 1)]
                if not any(is_improved):
                    print('BLEU score is not improved. Early stopping...')
                    break

        fp_loss.close()
        fp_bleu.close()
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            bleu, accuracy, hypotheses = self.evaluater(self.model, self.sources_test,
                                                        self.targets_test)
        self.hypotheses = hypotheses
        best_model.save_model(os.path.join(save_path, 'best.model'), suffix=False)


with open('./dataset/player_list.json') as f:
    id_to_player = json.load(f)

with open('./dataset/team_list.json') as f:
    id_to_team = json.load(f)


def convert(ind, id_to_word):
    if 'player' in ind:
        # i = ind.replace('player', '')
        # return id_to_player.get(i, ind)
        return ind
    elif 'team' in ind:
        # i = ind.replace('team', '')
        # return id_to_team.get(i, ind)
        return ind
    else:
        i = int(ind)
        if len(id_to_word) > i:
            return id_to_word[i]
        else:
            return ind


def masking(w):
    if 'player' in w:
        return 'player'
    elif 'team' in w:
        return 'team'
    else:
        return w


def evaluate_bleu(model, sources, targets):
    hypotheses = []
    references = []
    with chainer.using_config('train', False):
        for source, target in zip(sources, targets):
            y_batch = model.inference(source)
            hypotheses.extend(y_batch)
            references.extend([[y] for y in target])
    id_to_word = model.id_to_word
    if model.id_to_player:
        references = [[' '.join(y[0]).split()] for y in references]
    else:
        references = [[' '.join([convert(w, id_to_word) for w in y[0]]).split()]
                      for y in references]
    hypotheses = [' '.join([convert(w, id_to_word) for w in y]).split()
                  for y in hypotheses]
    print(' '.join(references[0][0]))
    print(' '.join(hypotheses[0]))
    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1) * 100

    return bleu, hypotheses


def evaluate_bleu_and_accuracy(model, sources, targets):
    hypotheses = []
    references = []
    pos_accuracy = 0
    neg_accuracy = 0
    data_length = 0
    with chainer.using_config('train', False):
        for source, target in zip(sources, targets):
            y_batch, prediction = model.inference(source)
            hypotheses.extend(y_batch)
            text, label, *_ = zip(*target)
            references.extend(text)
            label = chainer.Variable(cp.array(label, dtype=cp.int32))
            batch_size = label.shape[0]
            accuracy = binary_each_accuracy(prediction, label)
            pos_accuracy += accuracy[0] * batch_size
            neg_accuracy += accuracy[1] * batch_size
            data_length += batch_size
    pos_accuracy /= data_length
    neg_accuracy /= data_length
    id_to_word = model.id_to_word
    references = [[' '.join(y).split()] for y in references]
    hypotheses = [' '.join([convert(w, id_to_word) for w in y]).split()
                  for y in hypotheses]
    print(' '.join(references[0][0]))
    print(' '.join(hypotheses[0]))
    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1) * 100

    references_player = [
        [flatten([id_to_player.get(w.replace('player', ''), w).split() if 'player' in w else w for w in y[0]])]
        for y in references]
    hypotheses_player = [flatten([id_to_player.get(w.replace('player', ''), w).split()
                                  if 'player' in w else w for w in y]) for y in hypotheses]
    print(corpus_bleu(references_player, hypotheses_player, smoothing_function=SmoothingFunction().method1) * 100)

    def is_include(r):
        for w in r:
            if 'replace' in w or 'book' in w:
                return True
        return False
    ref, hyp = zip(*[(r, h) for r, h in zip(references, hypotheses) if not is_include(r[0])])
    print(corpus_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1) * 100)
    if model.tag_loss:
        for s in sources:
            print([model.id_to_player[int(x[0, 1].data)] for x in s])
            break
    return bleu, (pos_accuracy, neg_accuracy), hypotheses


def dump_setting(setting, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(setting, f)


def load_setting(filename):
    with open(filename) as f:
        text = f.read()
    return yaml.load(text)
