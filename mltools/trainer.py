import os
from collections import deque

import chainer
from tqdm import tqdm

builtin_print = print


def generate_file_writer(fp):
    def writer(text):
        builtin_print(text, file=fp, flush=True)
    return writer


def print(x): tqdm.write(str(x))


class ClassifierTrainer:
    def __init__(self, model, optimizer, x_train, y_train, x_test, y_test,
                 order_provider, epoch):
        self.model = model
        self.optimizer = optimizer
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.order_provider = order_provider
        self.data_length_train = len(x_train)
        self.data_length_test = len(x_test)
        self.train_steps = self.data_length_train // x_train.batch_size
        self.test_steps = self.data_length_test // x_test.batch_size
        self.epoch = epoch

    def train_one_epoch(self):
        sum_loss = 0
        for x_batch, y_batch in zip(tqdm(self.x_train, desc='train',
                                    total=self.train_steps), self.y_train):
            self.model.cleargrads()
            loss = self.model.loss(x_batch, y_batch)
            loss.backward()
            self.optimizer.update()
            sum_loss += loss.data * y_batch.shape[0]
            del loss
        self.order_provider.update()
        return sum_loss / self.data_length_train

    def test_one_epoch(self):
        sum_accuracy = 0
        for x_batch, y_batch in zip(tqdm(self.x_test, desc='test',
                                    total=self.test_steps), self.y_test):
            accuracy = self.model.accuracy(x_batch, y_batch)
            sum_accuracy += accuracy.data * y_batch.shape[0]
        return sum_accuracy / self.data_length_test

    def run(self):
        for i in tqdm(range(self.epoch), desc='epoch'):
            epoch = i + 1
            print('epoch: {}'.format(epoch))
            with chainer.using_config('train', True):
                average_loss = self.train_one_epoch()
            print('loss: {}'.format(average_loss))
            with chainer.using_config('train', False):
                average_accuracy = self.test_one_epoch()
            print('accuracy: {}'.format(average_accuracy))


class Seq2SeqTrainer:
    def __init__(self, model, optimizer, sources_train, targets_train,
                 sources_test, targets_test, order_provider,
                 evaluater, epoch, save_path, evaluation_step=1,
                 sources_train2=None, targets_train2=None, early_stopping=True):
        self.model = model
        self.optimizer = optimizer
        self.sources_train = sources_train
        self.targets_train = targets_train
        self.sources_test = sources_test
        self.targets_test = targets_test
        self.sources_train2 = sources_train2
        self.targets_train2 = targets_train2
        self.order_provider = order_provider
        self.evaluater = evaluater
        self.data_length = len(sources_train)
        self.epoch = epoch
        self.save_path = save_path
        self.evaluation_step = evaluation_step
        self.early_stopping = early_stopping

    def train_one_epoch(self):
        sum_loss = 0
        for x_batch, t_batch in zip(tqdm(self.sources_train, desc='train'),
                                    self.targets_train):
            self.model.cleargrads()
            loss, y_batch = self.model.loss(x_batch, t_batch)
            loss.backward()
            self.optimizer.update()
            sentence_length = len(t_batch)
            batch_size = t_batch[0].shape[0]
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
                    bleu, hypotheses = self.evaluater(self.model, self.sources_train2,
                                                      self.targets_train2)
                print('train BLEU score: {}'.format(bleu))
                with chainer.using_config('train', False), \
                        chainer.using_config('enable_backprop', False):
                    bleu, hypotheses = self.evaluater(self.model, self.sources_test,
                                                      self.targets_test)
                print('dev BLEU score: {}'.format(bleu))
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
            bleu, hypotheses = self.evaluater(self.model, self.sources_test,
                                              self.targets_test)
        self.hypotheses = hypotheses
        best_model.save_model(os.path.join(save_path, 'best.model'), suffix=False)
