import os
import csv
import re
import ast
import random
import statistics
from glob import iglob
from math import ceil
from itertools import chain, groupby
from collections import Counter
from unidecode import unidecode
import xml.etree.ElementTree as ET

import chainer
from chainer import functions as F
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from mltools.preprocessing import remove_ignore_chars, String2Tensor
from mltools.vocabulary import build_vocabulary
from mltools.data import Field, Loader, Dataset
from mltools.iterator import SequentialIterator
from config import event_type_mapper, qualifier_type_mapper
from utils import classproperty


class TextField(Field):

    home_player_tag = '<player>'
    away_player_tag = '<player>'
    home_team_tag = '<team>'
    away_team_tag = '<team>'
    # actions = ('clearance', 'shoot', 'block', 'save', 'shot',
    #            'book', 'replace', 'change', 'goal', '<other>')
    # actions = ('save from', 'the ball', 'makes', 'free-kick', 'dribbles',
    #            'shoots', 'goes close', '<other>')
    actions = ('save from', 'fires', 'cross', 'free-kick', 'forced', 'caught',
               'release', 'shot', 'chance', 'dribbles', 'shoots', 'goes close', '<other>')
    # actions = ('<other>',)
    # phrases = (['chance', 'for'], ['save', 'from'], ['is', 'forced'], ['is', 'met'],
    #            ['is', 'replaced'], ['is', 'booked'], ['play', 'from'], ['is', 'a'],
    #            ['is', 'denied'], ['shot', 'from'], ['the', 'ball'], ['is', 'the'],
    #            ['is', 'down'], ['is', 'credited'], ['is', 'blocked'], ['has', 'been'],
    #            ['is', 'shown'], ['goes', 'close'], ['is', 'caught'])
    phrases = (
        ['of', 'the'], ['the', 'ball'], ['into', 'the'], ['on', 'the'], ['in', 'the'],
        ['down', 'the'], ['from', 'the'], ['as', 'he'], ['for', 'the'], ['the', 'left'],
        ['the', 'right'], ['to', 'the'], ['at', 'the'], ['but', 'the'], ['the', 'box'],
        ['the', 'area'], ['penalty', 'area'], ['edge', 'of'], ['the', 'edge'], ['with', 'the'],
        ['over', 'the'], ['to', 'get'], ['the', 'game'], ['it', "'s"], ['replaced', 'by'],
        ['well', 'to'], ['but', 'he'], ['west', 'ham'], ['the', 'pitch'], ['to', 'be'],
        ['and', 'the'], ['the', 'back'], ['as', 'the'], ['yellow', 'card'], ['the', 'visitors'],
        ['the', 'hosts'], ['inside', 'the'], ['tries', 'to'], ['the', 'first'], ['the', 'penalty'],
        ['up', 'the'], ['but', 'it'], ['off', 'the'], ['towards', 'the'], ['free', 'kick'],
        ['out', 'of'], ['by', 'the'], ['picks', 'up'], ['the', 'referee'], ['but', 'his'],
        ['west', 'brom'], ['is', 'replaced'], ['has', 'been'], ['and', 'it'], ['does', 'well'],
        ['and', 'he'], ['area', 'but'], ['cross', 'into'], ['his', 'way'], ['the', 'danger'],
        ['the', 'middle'], ['away', 'from'], ['to', 'make'], ['chance', 'for'], ['cross', 'from'],
        ['area', 'and'], ['the', 'far'], ['who', 'has'], ['ball', 'into'], ['the', 'final'],
        ['able', 'to'], ['once', 'again'], ['ball', 'in'], ['this', 'time'], ['right', 'wing'],
        ['and', 'is'], ['for', 'corner'], ['left', 'wing'], ['box', 'but'], ['the', 'right-hand'],
        ['it', 'is'], ['the', 'top'], ['on', 'to'], ['left', 'flank'], ['near', 'post'],
        ['the', 'striker'], ['wide', 'of'], ['is', 'booked'], ['right', 'flank'], ['out', 'for'],
        ['right-hand', 'side'], ['the', 'left-hand'], ['the', 'home'], ['back', 'post'],
        ['middle', 'of'], ['the', 'end'], ['the', 'bar'], ['with', 'his'], ['looks', 'to'],
        ['out', 'to'], ['left-hand', 'side'], ['ca', "n't"], ['on', 'for'], ['the', 'path'],
        ['path', 'of'], ['of', 'play'], ['he', "'s"], ['makes', 'his'], ['the', 'near'],
        ['far', 'post'], ['the', 'corner'], ['the', 'former'], ['unable', 'to'], ['booked', 'for'],
        ['who', 'is'], ['in', 'from'], ['is', 'the'], ['to', 'find'], ['comes', 'on'],
        ['change', 'of'], ['place', 'of'], ['back', 'to'], ['play', 'from'], ['end', 'of'],
        ['in', 'behind'], ['manages', 'to'], ['the', 'second'], ['he', 'is'], ['now', 'as'],
        ['for', 'his'], ['way', 'for'], ['the', 'break'], ['box', 'and'], ['yards', 'out'],
        ['low', 'cross'], ['his', 'shot'], ['area', 'before'], ['his', 'effort'], ['side', 'of'],
        ['before', 'the'], ['pass', 'from'], ['on', 'goal'], ['to', 'his'], ['in', 'place'],
        ['of', 'space'], ['de', 'bruyne'], ['the', 'match'], ['through', 'the'], ['foul', 'on'],
        ['to', 'pick'], ['card', 'for'], ['pass', 'into'], ['now', 'for'], ['to', 'clear'],
        ['so', 'far'], ['fails', 'to'], ['be', 'replaced'], ['the', 'west'], ['as', 'they'],
        ['cross', 'towards'], ['the', 'crossbar'], ['cross', 'is'], ['of', 'his'], ['goal', 'but'],
        ['ball', 'out'], ['trying', 'to'], ['for', 'west'], ['the', 'target'], ['have', 'been'],
        ['the', 'post'], ['the', 'net'], ["'s", 'cross'], ['behind', 'the'], ['on', 'his'],
        ['change', 'for'], ['final', 'change'], ['shot', 'from'], ['to', 'take'], ['ends', 'up'],
        ['goes', 'down'], ['resulting', 'in'], ['it', 'was'], ['his', 'own'], ['home', 'side'],
        ['bit', 'of'], ['corner', 'but'], ['to', 'break'], ['side', 'but'], ['to', 'have'],
        ['ball', 'and'], ['pick', 'out'], ['challenge', 'on'], ['the', 'ground'], ['his', 'first'],
        ['just', 'outside'], ['the', 'spaniard'], ['for', 'goal'], ['left', 'and'], ['post', 'but'])

    team_nicknames = ['spurs']

    def __init__(self, start_token='<s>', end_token='</s>', fix_length=None, ignore_label=-1,
                 mask_player=False, mask_team=False, numbering=False, reverse=True,
                 bpc=False, multi_tag=False):
        self.start = [start_token] if start_token else []
        self.end = [end_token] if end_token else []
        self.ignore_label = ignore_label
        self.mask_player = mask_player
        self.mask_team = mask_team
        self.mask_mode = mask_player or mask_team
        self.numbering = numbering
        self.bpc = bpc
        self.multi_tag = multi_tag
        self._player_num = 0
        self._team_num = 0
        self.reverse = reverse
        if fix_length is not None:
            self.fix_length = slice(0, fix_length - len(self.start + self.end))
        else:
            self.fix_length = slice(0, None)
        super().__init__()

    def numericalize(self, example):
        unk_id = self.unk_id
        return [str(self.word_to_id.get(x, unk_id)) for x in example]

    def preprocess(self, example):
        x = self._tokenize(example)
        if self.bpc:
            x = self.merge_words(x)
        if self.mask_mode:
            x = self._mask(example, x)
        x = self.start + self.cut_sentence(x) + self.end
        x = x[::-1] if self.reverse else x
        if self.numbering:
            x = self._numbering(x)
        return x

    def cut_sentence(self, text):
        temp = text[self.fix_length]
        if self.home_player_tag in temp or self.away_player_tag in temp:
            return temp
        elif self.home_player_tag in text or self.away_player_tag in text:
            i = text.index('<player>')
            step = self.fix_length.stop - self.fix_length.start
            start = i - step if i - step > 0 else 0
            stop = start + step
            return text[start:stop]
        return temp

    def merge_words(self, text):
        for phrase in self.phrases:
            len_phrase = len(phrase)
            for i in range(len(text) - len_phrase + 1):
                if phrase == text[i:i + len_phrase]:
                    for _ in range(len_phrase - 1):
                        text[i] += ' {}'.format(text.pop(i + 1))
        return text

    def _numbering(self, x):
        text_type = '<other>'
        if self.multi_tag:
            text = ' '.join(x)
            for action in self.actions:
                if action in text:
                    text_type = action
                    break
        x_new = []
        i_player = i_team = 0
        for w in x:
            if w == '<player>':
                w = w + text_type + str(i_player)
                i_player += 1
            elif w == '<team>':
                w = w + str(i_team)
                i_team += 1
            x_new.append(w)
        self._player_num = max(i_player, self._player_num)
        self._team_num = max(i_team, self._team_num)
        return x_new

    def _mask(self, example, x, keep_id=False):
        if keep_id:
            home_players = example['home_players']
            away_players = example['away_players']
            for names, player_id in list(home_players.items()) + list(away_players.items()):
                new = 'player{}'.format(player_id)
                for name in [n.lower() for n in names[1:-1].split('|')]:
                    if name in x:
                        x = [w.replace(name, new) if w == name else w for w in x]
            teams = {example['home_team_id']: example['home_team_name'].lower().split(' '),
                     example['away_team_id']: example['away_team_name'].lower().split(' ')}
            for team_id, team in teams.items():
                new = 'team{}'.format(team_id)
                for t in team:
                    if t in x:
                        x = [w.replace(t, new) if w == t else w for w in x]
        else:
            home_players = list(example['home_players'].keys())
            away_players = list(example['away_players'].keys())
            if not self.mask_mode:
                return x
            x = self._replace_players(x, home_players, self.home_player_tag)
            x = self._replace_players(x, away_players, self.away_player_tag)
            if not self.mask_team:
                x = [w for w, _ in groupby(x)]
                return x
            home_team = example['home_team_name'].lower().split(' ')
            away_team = example['away_team_name'].lower().split(' ')
            x = self._replace_team(x, home_team + self.team_nicknames, self.home_team_tag)
            x = self._replace_team(x, away_team + self.team_nicknames, self.away_team_tag)
        x = [w for w, _ in groupby(x)]
        return x

    @staticmethod
    def _tokenize(example):
        sentences = sent_tokenize(example['comment'])
        players = list(example['home_players'].keys()) + list(example['away_players'].keys())
        target = 100  # just big number
        for i, s in enumerate(sentences):
            is_exists = [bool(re.search(pat, s)) for pat in players]
            if any(is_exists):
                target = min(i, target)
            if all(is_exists):
                target = i
                break
        return word_tokenize(remove_ignore_chars(sentences[target].lower()))

    @staticmethod
    def _replace_players(x, players, tag):
        for p in players:
            for name in p[1:-1].split('|'):
                names = [n.lower() for n in name.split(' ')]
                for n in names:
                    if n in x:
                        x = [w.replace(n, tag) if w == n else w for w in x]
        return x

    @staticmethod
    def _replace_team(x, teams, tag):
        for team in teams:
            if team in x:
                x = [w.replace(team, tag) if w == team else w for w in x]
        return x

    def build_vocabulary(self, examples, size=None,
                         special_words=('<unk>', '<s>', '</s>'), set_as_property=True):
        counter = Counter(chain.from_iterable(examples))
        count_pairs = counter.most_common()

        words, _ = list(zip(*count_pairs))
        if special_words:
            words = special_words + words
        vocab_size = len(words)
        assigned_size = size or vocab_size
        size = assigned_size if assigned_size < vocab_size else vocab_size
        indices = range(size)
        word_to_id = dict(zip(words, indices))
        if set_as_property:
            self.word_to_id = word_to_id
            self.id_to_word = words[:size]
        self.unk_id = word_to_id['<unk>']
        if getattr(self, 'numbering', None) and \
                not getattr(self, 'player_id', None) and not getattr(self, 'team_id', None):
            for action in self.actions:
                self.player_id = [word_to_id['<player>{}{}'.format(action, i)]
                                  for i in range(self._player_num)
                                  if '<player>{}{}'.format(action, i) in word_to_id]
            self.team_id = [word_to_id['<team>{}'.format(i)]
                            for i in range(self._team_num)
                            if '<team>{}'.format(i) in word_to_id]
        return word_to_id, words[:size]

    def compute_average_length(self, examples):
        length_data = [len(ex) for ex in examples]
        return statistics.mean(length_data), statistics.variance(length_data)


class TextAndContentWordField(TextField):

    content_tag = ('NN', 'VB', 'JJ', 'RB')
    player_pat = re.compile('player\d+')
    team_pat = re.compile('team\d+')

    def preprocess(self, example):
        text = super()._tokenize(example)
        content_words = [word for word, tag in nltk.pos_tag(text)
                         if any(t in tag for t in self.content_tag)]
        if self.bpc:
            text = self.merge_words(text)
        masked_text = super()._mask(example, text, True)
        player_id = [int(w.replace('player', '')) if self.player_pat.match(w) is not None else -1
                     for w in masked_text]
        team_id = [int(w.replace('team', '')) if self.team_pat.match(w) is not None else -1
                   for w in masked_text]
        if self.mask_mode:
            text = super()._mask(example, text)
        text = text[::-1] if self.reverse else text
        text = self.start + self.cut_sentence(text) + self.end
        if self.numbering:
            text = self._numbering(text)
        return text, content_words, player_id, team_id

    def build_vocabulary(self, examples, size=None,
                         special_words=('<unk>', '<s>', '</s>')):
        text, content_word, player_id, team_id = zip(*examples)
        word_to_id, words = super().build_vocabulary(text, size, special_words, True)
        content_word_to_id, content_words = super().build_vocabulary(
            content_word, None, ('<unk>',), False)
        player_id = [[i for i in x if i != -1] for x in player_id]
        player_to_id, players = super().build_vocabulary(player_id, None, ('<unk>',), False)
        team_id = [[i for i in x if i != -1] for x in team_id]
        team_to_id, teams = super().build_vocabulary(team_id, None, ('<unk>',), False)
        self.content_word_to_id = content_word_to_id
        self.content_words = content_words
        self.player_to_id = player_to_id
        self.players = players
        self.team_to_id = team_to_id
        self.teams = teams
        return word_to_id, words, content_word_to_id, content_words

    def numericalize(self, example):
        if len(example) == 2:
            text, content_word = example
        elif len(example) == 4:
            text, content_word, player_id, team_id = example
            tensor_player_id = [self.player_to_id.get(i) if i != -1 else i for i in player_id]
            tensor_team_id = [self.team_to_id.get(i) if i != -1 else i for i in team_id]
        tensor_text = super().numericalize(text)
        unk_id = self.content_word_to_id['<unk>']
        tensor_content_word = [0] * len(self.content_word_to_id)
        for w in content_word:
            tensor_content_word[self.content_word_to_id.get(w, unk_id)] = 1
        if len(example) == 2:
            return tensor_text, tensor_content_word
        elif len(example) == 4:
            return tensor_text, tensor_content_word, tensor_player_id, tensor_team_id


class TestTextField(TextAndContentWordField):
    def __init__(self, id_to_player, id_to_team, word_to_id, content_word_to_id=None,
                 unk_id=0, fix_length=None, ignore_label=-1, mask_player=True, mask_team=True,
                 bpc=False):
        super().__init__(None, None, fix_length, ignore_label, mask_player, mask_team,
                         bpc=bpc)
        self.id_to_player = id_to_player
        self.id_to_team = id_to_team
        self.word_to_id = word_to_id
        self.content_word_to_id = content_word_to_id
        self.unk_id = unk_id

    def numericalize(self, data):
        if self.content_word_to_id:
            x, content_words, example = data
        else:
            x, example = data
        x = ' '.join(x)
        x = self._replace_players(x, example['home_players'])
        x = self._replace_players(x, example['away_players'])
        home_team = example['home_team_name'].lower().split(' ')
        away_team = example['away_team_name'].lower().split(' ')
        x = self._replace_team(x, home_team + self.team_nicknames, example['home_team_id'])
        x = self._replace_team(x, away_team + self.team_nicknames, example['away_team_id'])
        x = x.split()
        # if self.bpc:
        #     x = self.merge_words(x)
        # unk_id = self.unk_id
        # x = [w if 'player' in w or 'team' in w else str(self.word_to_id.get(w, unk_id))
        #      for w, _ in groupby(self.start + x[self.fix_length] + self.end)]
        x = [w if 'player' in w or 'team' in w else w
             for w, _ in groupby(self.start + x[self.fix_length] + self.end)]
        if self.content_word_to_id:
            _, tensor_content_word = super().numericalize((x, content_words))
            return x, tensor_content_word
        else:
            return x

    def preprocess(self, example):
        x = super()._tokenize(example)
        if self.content_word_to_id:
            content_words = [word for word, tag in nltk.pos_tag(x)
                             if any(t in tag for t in self.content_tag)]
        if self.content_word_to_id:
            return x, content_words, example
        else:
            return x, example

    @staticmethod
    def _replace_players(sentence, players, tag='player{}'):
        for p, i in players.items():
            for name in p[1:-1].split('|'):
                names = [n.lower() for n in name.split(' ')]
                for n in names:
                    if n in sentence:
                        sentence = sentence.replace(n, tag.format(i))
        return sentence

    @staticmethod
    def _replace_team(sentence, teams, team_id, tag='team{}'):
        for team in teams:
            if team in sentence:
                sentence = sentence.replace(team, tag.format(team_id))
        return sentence


class EventField(TextField):
    def __init__(self, ignore_label=-1, fix_length=None, embed_size=None):
        self.ignore_label = ignore_label
        self.fix_length = fix_length
        self._details_dimention = 0
        self.embed_size = embed_size
        super(TextField, self).__init__()

    @property
    def fillvalue(self):
        return [-1] * 9 + [-1] * self.details_dimention

    @property
    def attention_size(self):
        return [(i, self.embed_size) for i in range(3)] + [(3, 6)] +\
            [(i, self.embed_size) for i in range(self.details_dimention)]

    @staticmethod
    def preprocess(example):
        return [[x['type_id'], x['player_id'], x['team_id'], x['outcome'], x['x'],
                 x['y'], x['end_x'], x['end_y'], x['delta'], [tuple(i) for i in x['details']]]
                for x in example]

    @property
    def details_dimention(self):
        return self._details_dimention

    def numericalize(self, example):
        details_dimention = self.details_dimention
        detail_unk_id = self.detail_unk_id

        def _convert(x):
            detail_vec = [0] * details_dimention
            for i in x[9]:
                detail_vec[self.detail_to_id.get(i, detail_unk_id)] = 1.
            return [self.type_to_id.get(x[0], self.type_unk_id),
                    self.player_to_id.get(x[1], self.player_unk_id),
                    self.team_to_id.get(x[2], self.team_unk_id), *x[3:9]] + \
                detail_vec
            # return [self.type_to_id.get(x[0], self.type_unk_id),
            #         self.player_to_id.get(x[1], self.player_unk_id),
            #         self.team_to_id.get(x[2], self.team_unk_id), *x[3:9],
            #         [self.detail_to_id.get(i, self.detail_unk_id) for i in x[9]]]
        example = [_convert(x) for x in example[:self.fix_length]]
        # ignore_label = self.ignore_label
        # details_dimention = self.details_dimention
        # return [x[:9] + x[9] + [ignore_label] * (details_dimention - len(x[9]))
        #         for x in example]
        return example

    def build_vocabulary(self, examples, size=None, special_words=None):
        type_ids, player_ids, team_ids, _, _, _, _, _, _, details = \
            zip(*chain.from_iterable(examples))

        def remove_minus_one(ids): return [i for i in ids if i != -1]
        type_ids = remove_minus_one(type_ids)
        player_ids = remove_minus_one(player_ids)
        team_ids = remove_minus_one(team_ids)

        func = super().build_vocabulary
        special_words = ('<unk>',)
        self.type_to_id, self.id_to_type = func([type_ids], None, special_words)
        self.player_to_id, self.id_to_player = func([player_ids], None, special_words)
        self.team_to_id, self.id_to_team = func([team_ids], None, special_words)
        self.detail_to_id, self.id_to_detail = func(details, None,
                                                    special_words)
        self.type_unk_id = self.type_to_id['<unk>']
        self.player_unk_id = self.player_to_id['<unk>']
        self.team_unk_id = self.team_to_id['<unk>']
        self.detail_unk_id = self.detail_to_id['<unk>']
        # self._details_dimention = max(len(x) for x in details)
        self._details_dimention = len(self.detail_to_id)
        del self.word_to_id
        del self.id_to_word


class OptaDataset(Dataset):
    def __init__(self, path='./dataset/parallel.json',
                 fields={'source': EventField(), 'target': TextField()},
                 limit_length=None):
        examples = Loader.from_json(path, fields)
        if limit_length is not None:
            flags = []
            for i, trg in enumerate(examples['target']):
                if type(trg) == tuple:
                    trg = trg[0]
                flags.append(len(trg) <= limit_length)
            examples['source'].data = [v for i, v in enumerate(examples['source'].data)
                                       if flags[i]]
            examples['target'].data = [v for i, v in enumerate(examples['target'].data)
                                       if flags[i]]
        super().__init__(examples, fields)


class TextAndLabelIterator(SequentialIterator):
    def _wrapper(self, batch):
        text, label, player, team = zip(*batch)
        text = super()._wrapper(text)
        label = chainer.Variable(self.xp.asarray(label, dtype=self.xp.int32))
        player = F.transpose_sequence(self.xp.asarray(self.padding(player), dtype=self.xp.int32))
        team = F.transpose_sequence(self.xp.asarray(self.padding(team), dtype=self.xp.int32))
        return text, label, player, team

    def padding(self, batch):
        max_length = max(len(x) for x in batch)
        return [x + [-1] * (max_length - len(x)) for x in batch]


class Games:
    def __init__(self):
        self.games = []

    def add_game(self, game):
        if game.game_id is None:
            raise Exception
        self.games.append(game)

    def get_inputs(self, event_size=10, time_threshold=240):
        def key_func(e): return e.get_second()

        def _get_input():
            for game in self.games:
                for es, m in game.provide_aligned_event_message(time_threshold):
                    source_input = [e.to_vector() for e in
                                    sorted(es[:event_size], key=key_func)]
                    length = len(source_input)
                    if length < event_size:
                        source_input += [Event.fill_value
                                         for _ in range(event_size - length)]
                    target_input = m.comment
                    yield source_input, target_input
        source_inputs, target_inputs = zip(*((s, t) for s, t in _get_input()))
        return source_inputs, target_inputs

    def get_strict_aligned_inputs(self):
        def _get_input():
            for game in self.games:
                for es, m in game.provide_strict_aligned_event_message():
                    yield [e.to_vector() for e in es], m.comment

        def _padding(x):
            return x + [Event.fill_value for _ in range(max_length - len(x))]
        source_inputs, target_inputs = zip(*((s, t) for s, t in _get_input()))
        max_length = max(len(x) for x in source_inputs)
        return [_padding(x) for x in source_inputs], target_inputs

    @staticmethod
    def preprocess_target_inputs(target_inputs):
        # lower -> remove some symbol -> word tokenize
        return [word_tokenize(remove_ignore_chars(y.lower()))
                for y in target_inputs]

    @staticmethod
    def compute_source_vocabularies(source_inputs, unk_token='<unk>'):
        def remove_minus_one(ids): return [i for i in ids if i != -1]
        type_ids, player_ids, team_ids, _, _, _, _, _, details = \
            zip(*chain.from_iterable(source_inputs))
        type_ids = remove_minus_one(type_ids)
        player_ids = remove_minus_one(player_ids)
        team_ids = remove_minus_one(team_ids)
        special_words = (unk_token,)
        type_to_id, id_to_type = build_vocabulary(type_ids, special_words=special_words)
        player_to_id, id_to_player = build_vocabulary(player_ids, special_words=special_words)
        team_to_id, id_to_team = build_vocabulary(team_ids, special_words=special_words)
        detail_to_id, id_to_detail = build_vocabulary(chain.from_iterable(details),
                                                      special_words=special_words)
        type_to_id[-1] = -1
        player_to_id[-1] = -1
        team_to_id[-1] = -1
        detail_to_id[-1] = -1
        vocab_dict = {'type_to_id': type_to_id, 'id_to_type': id_to_type,
                      'player_to_id': player_to_id, 'id_to_player': id_to_player,
                      'team_to_id': team_to_id, 'id_to_team': id_to_team,
                      'detail_to_id': detail_to_id, 'id_to_detail': id_to_detail}
        return vocab_dict

    @staticmethod
    def compute_target_vocabulary(target_inputs):
        word_to_id, id_to_token = build_vocabulary(
            chain.from_iterable(target_inputs))
        return word_to_id, id_to_token

    @staticmethod
    def transform_source_inputs(source_inputs, vocab_dict, ignore_label=-1):
        max_length = 0

        def _transform(x):
            nonlocal max_length
            max_length = max(max_length, len(x[-1]))
            return [vocab_dict['type_to_id'][x[0]], vocab_dict['player_to_id'][x[1]],
                    vocab_dict['team_to_id'][x[2]], *x[3:8],
                    [vocab_dict['detail_to_id'][i] for i in x[8]]]

        def _padding(x):
            return x[:8] + x[8] + [ignore_label] * (max_length - len(x[8]))
        transformed = [[_transform(x) for x in xs] for xs in source_inputs]
        return [[_padding(x) for x in xs] for xs in transformed], max_length

    @staticmethod
    def transform_target_inputs(target_inputs, word_to_id, unk_id, start_id,
                                end_id, sentence_size):
        s2t = String2Tensor(word_to_id, unk_id, start_id, end_id, sentence_size)
        return [s2t.encode(y) for y in target_inputs]

    @staticmethod
    def shuffle_inputs(*inputs, seed=None):
        length = min(len(x) for x in inputs)
        random.seed(seed)
        return zip(*random.sample(list(zip(*inputs)), length))

    @staticmethod
    def split_inputs(*inputs, **options):
        length = min(len(x) for x in inputs)
        train_index_end = ceil(length * options.get('train_size', 0.8))
        test_index_end = train_index_end + ceil(length * options.get('test_size', 0.1))
        train_slicer = slice(0, train_index_end)
        test_slicer = slice(train_index_end, test_index_end)
        dev_slicer = slice(test_index_end, length)
        return list(chain.from_iterable([
            (x[train_slicer], x[test_slicer], x[dev_slicer]) for x in inputs]))

    def dump_info(self):
        import numpy as np
        source_inputs, target_inputs, type_vocab, player_vocab, team_vocab,\
            id_to_word, word_to_id = self.make_inputs(split=False, debug_size=-1)
        print('--- source data ---')
        print('longest sequence: {}'.format(max(len(x) for x in source_inputs)))
        print('shortest sequence: {}'.format(min(len(x) for x in source_inputs)))
        print('type vocab: {}'.format(len(type_vocab)))
        print('player vocab: {}'.format(len(player_vocab)))
        print('--- target data ---')
        print('longest sequence: {}'.format(max(len(x) for x in target_inputs)))
        print('shortest sequence: {}'.format(min(len(x) for x in target_inputs)))
        print('vocab size: {}'.format(len(id_to_word)))
        print('--- game data ---')
        game_sum = len(self.games)
        print('sum of games: {}'.format(game_sum))
        print('--- event data ---')
        event_len = [len(g.events) for g in self.games]
        print('sum of events: {}'.format(sum(event_len)))
        print('average event: {}'.format(np.mean(event_len)))
        print('variance event: {}'.format(np.var(event_len)))
        print('--- message data ---')
        message_len = [len(g.messages) for g in self.games]
        print('sum of messages: {}'.format(sum(message_len)))
        print('average message: {}'.format(np.mean(message_len)))
        print('variance message: {}'.format(np.var(message_len)))

    def output_file(self, target_inputs_train, target_inputs_test, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        def writer(filename, data):
            with open(filename, 'w', encoding='utf-8') as fp:
                [fp.write(' '.join([x for x in xs]) + '\n') for xs in data]
        writer(os.path.join(save_path, 'training.txt'), target_inputs_train)
        writer(os.path.join(save_path, 'references.txt'), target_inputs_test)

    def __getitem__(self, key):
        return self.games[key]

    def __setitem__(self, key, value):
        self.games[key] = value


class Game:
    def __init__(self):
        self.game_id = None
        self.messages = []
        self.events = []

    def set_game_id(self, game_id):
        self.game_id = game_id

    def add_message(self, message):
        self.messages.append(message)

    def add_event(self, event):
        self.events.append(event)

    def display_messages(self):
        base_message = '{}:{} : {}'
        for m in self.messages:
            print(base_message.format(m.minute, m.second, m.comment))

    def display_events(self, verbose=False):
        event_message = '{}:{} : {}'
        qualifier_message = '   {}'
        for e in self.events:
            print(event_message.format(e.minute, e.second, event_type_mapper[e.type_id]))
            if verbose:
                for q in e.qualifiers:
                    print(qualifier_message.format(qualifier_type_mapper[q.qualifier_id]))

    def provide_aligned_event_message(self, threshold=300):
        self._filter_events()
        self._filter_messages()

        def time_diff(e, m): return abs(e.get_second() - m.get_second())

        def align(events, message):
            players = message.players
            # remove events less than threshold
            events = [(e, time_diff(e, message)) for e in events
                      if time_diff(e, message) < threshold]
            # sort by time
            events, _ = zip(*sorted(events, key=lambda x: x[1]))
            if players:
                # sort by player
                events = sorted(events, key=lambda e: e.player_id in players)
            return events
        for m in self.messages:
            yield align(self.events, m), m

    def provide_strict_aligned_event_message(self):
        self._filter_events()
        self._filter_messages()

        def time_diff(e, m): return abs(e.get_second() - m.get_second())

        def align(events, message):
            players = message.players
            # sort by time
            tmp, _ = zip(*sorted([(x, time_diff(x, m)) for x in events],
                                 key=lambda x: x[1]))
            events = []
            for player_id in players:
                for x in tmp:
                    if x.player_id == player_id:
                        events.append(x)
                        break
                if len(players) == len(events):
                    break
            return events
        for m in self.messages:
            if not m.players:
                continue
            yield align(self.events, m), m

    def _filter_events(self):
        events = self.events
        # filter 0:0 (this event shouldn't be included)
        events = [e for e in events if e.minute or e.second]
        # filter start and end event
        events = [e for e in events if e.type_id != 30 and e.type_id != 32]
        # filter post-game, pre-game, pre-match
        ignore_period = (14, 15, 16)
        events = [e for e in events if e.period_id not in ignore_period]
        self.events = events

    def _filter_messages(self):
        messages = self.messages
        # messages = [m for m in messages if m.minute or m.second]
        ignore_types = ('full time', 'half time', 'kick off', 'pre kick off',
                        'second half')
        messages = [m for m in messages if m.comment_type not in ignore_types]
        messages = [m for m in messages if m.time != '']
        self.messages = messages


class Event:

    def __init__(self, event_id, type_id, team_id, player_id, minute, second,
                 outcome, x, y, detail, end_x, end_y, period_id):
        self.event_id = int(event_id)
        self.type_id = int(type_id)
        self.team_id = int(team_id)
        self.player_id = int(player_id)
        self.minute = int(minute)
        self.second = int(second)
        self.outcome = int(outcome)
        self.x = float(x)
        self.y = float(y)
        # self.timestamp = timestamp
        self.detail = detail
        self.end_x = float(end_x)
        self.end_y = float(end_y)
        self.period_id = int(period_id)
        self.qualifiers = []

    def add_qualifier(self, qualifier):
        self.qualifiers.append(qualifier)

    def to_vector(self):
        return [self.type_id, self.player_id, self.team_id, self.outcome,
                self.x, self.y, self.end_x, self.end_y, self.detail]

    @classproperty
    def fill_value(cls):
        return [-1] * 8 + [[]]

    def get_second(self):
        return self.minute * 60 + self.second


class Qualifier:
    def __init__(self, qualifier_id, value):
        self.qualifier_id = int(qualifier_id)
        self.value = value


class Commentary:
    def __init__(self, home_score, away_score):
        self.home_score = home_score
        self.away_score = away_score
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_word_list(self):
        text = ''.join([m.comment for m in self.messages])
        return text


class Message:
    def __init__(self, comment_type, comment, minute, second, players,
                 period, time):
        self.comment_type = comment_type
        self.comment = comment
        self.minute = int(minute) if minute else 0
        self.second = int(second) if second else 0
        self.players = players
        self.period = period
        self.time = time

    def to_input(self):
        return self.comment

    def get_second(self):
        return self.minute * 60 + self.second


def load_dataset(f13_path='./dataset/F13M', f24_path='./dataset/F24',
                 gameid_file='./dataset/gameid.txt', debug_size=-1):
    games = Games()
    lines = gameid_fp = open(gameid_file)
    if debug_size > 0:
        lines = (next(gameid_fp) for _ in range(debug_size))
    for gameid in lines:
        game = Game()
        gameid = gameid.rstrip()
        f24_file = os.path.join(f24_path, 'F24_gameid{}.csv'.format(gameid))
        f13_file = os.path.join(f13_path, 'F13M_gameid{}.csv'.format(gameid))
        with open(f24_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            for line in reader:
                line[9] = ast.literal_eval(line[9])
                event = Event(*line)
                game.add_event(event)
        with open(f13_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            for line in reader:
                line[4] = ast.literal_eval(line[4])
                message = Message(*line)
                game.add_message(message)
        game.set_game_id(gameid)
        games.add_game(game)
    gameid_fp.close()
    return games


class PlayerToId(dict):
    def __getitem__(self, key):
        value = super().get(key, None)
        if value:
            return value
        if not getattr(self, 'player_names', None):
            self.set_player_names()
        for names in self.player_names:
            if key in names:
                return super().__getitem__(names)
        raise KeyError

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def set_player_names(self):
        self.player_names = self.keys()


def load_players(f42_file='/auto/local/data/corpora/Opta/F42/F42_competiiton8_seasonid2015.xml'):
    root = ET.parse(f42_file).getroot()
    teams = root.findall('.//Team')

    def to_ascii_str(string):
        return unidecode(string).encode('ascii').decode('utf-8')

    def filter_element(element):
        if element.tag == 'Name':
            return True
        elif element.tag == 'Stat':
            if 'name' in element.attrib['Type']:
                return True
        return False

    def get_key(player):
        return tuple(to_ascii_str(e.text)
                     for e in player.getchildren() if filter_element(e))

    def get_id(element):
        return int(element.attrib['uID'][1:])

    # these player id not exists in F42 file, so I added manually
    addtional_players = (
        (57, ('Alessandro Diamanti',  'Alessandro',  'Diamanti'), 45129),
        (13, ('Andrej Kramaric',  'Andrej',  'Kramaric'), 68582),
        (45, ('Bradley Johnson',  'Bradley',  'Johnson',  'Paul'), 19569),
        (45, ('Gary Hooper',  'Gary',  'Hooper'), 28221),
        (1, ('Javier Hernandez',  'Javier',  'Hernandez',  'Chicharito'), 43020),
        (21, ('Kevin Nolan',  'Kevin',  'Nolan',  'Anthony Jance'), 5306),
        (43, ('Patrick Roberts',  'Patrick',  'Roberts',  'John Joseph'), 124165),
        (57, ('Victor Ibarbo',  'Victor',  'Ibarbo',  'Segundo',  'Guerrero'), 59380),
        (91, ('Yann Kermorgant',  'Yann',  'Kermorgant',  'Alain'), 44558))
    team_id = {}
    for team in teams:
        players = team.findall('./Player')
        team_id[get_id(team)] = PlayerToId((get_key(p), get_id(p)) for p in players)
    for team, key, value in addtional_players:
        team_id[team][key] = value
    return team_id


def basename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]


def generate_message_dataset(f13m_path='/auto/local/data/corpora/Opta/F13M',
                             dest_path='./dataset/F13M'):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    f13m_files = iglob(os.path.join(f13m_path, '*.xml'))
    team_id = load_players()
    header = ['type', 'comment', 'minute', 'second', 'player_ids', 'period', 'time']
    for f13m_file in f13m_files:
        root = ET.parse(f13m_file).getroot()
        away_id = int(root.attrib['away_team_id'])
        home_id = int(root.attrib['home_team_id'])
        messages = reversed(root.findall('./message'))
        message_data = [header]
        for m in messages:
            attrib = m.attrib
            comment = attrib['comment']
            ids = []
            for w, pos in nltk.pos_tag(word_tokenize(comment)):
                if pos == 'NNP':
                    player1 = team_id[away_id].get(w, None)
                    player2 = team_id[home_id].get(w, None)
                    if player1 not in ids and player1 is not None:
                        ids.append(player1)
                    elif player2 not in ids and player2 is not None:
                        ids.append(player2)
            for x in sent_tokenize(comment):
                message_data.append([attrib['type'], x, attrib.get('minute'),
                                     attrib.get('second'), ids,
                                     attrib.get('period'), attrib['time']])
        csv_filename = '{}.csv'.format(basename_without_ext(f13m_file))
        with open(os.path.join(dest_path, csv_filename), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(message_data)


def generate_event_dataset(f24_path='/auto/local/data/corpora/Opta/F24',
                           dest_path='./dataset/F24'):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    f24_files = iglob(os.path.join(f24_path, '*.xml'))

    def parse(event):
        attrib = event.attrib
        event_id = int(attrib['event_id'])
        type_id = int(attrib['type_id'])
        team_id = int(attrib.get('team_id', -1))
        player_id = int(attrib.get('player_id', -1))
        period_id = int(attrib['period_id'])
        minute = int(attrib['min'])
        second = int(attrib['sec'])
        outcome = int(attrib['outcome'])
        x = float(attrib['x']) / 100
        y = float(attrib['y']) / 100
        qs = [(int(q.attrib['qualifier_id']), q.attrib.get('value'))
              for q in event.findall('./Q')]
        type_outcome_q = [(type_id, outcome, q[0]) for q in qs]
        end_xy = {'x' if q[0] == 140 else 'y': float(q[1]) / 100
                  for q in qs if q[0] in (140, 141)}
        return event_id, type_id, team_id, player_id, minute, second, outcome,\
            x, y, type_outcome_q, end_xy.get('x', -1), end_xy.get('y', -1), period_id
    header = ['event_id', 'type_id', 'team_id', 'player_id', 'minute', 'second',
              'outcome', 'x', 'y', 'detail', 'end_x', 'end_y']
    for f24_file in f24_files:
        root = ET.parse(f24_file).getroot()
        events = [header]
        for event in root.findall('.//Event'):
            events.append(parse(event))
        csv_filename = '{}.csv'.format(basename_without_ext(f24_file))
        with open(os.path.join(dest_path, csv_filename), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(events)


if __name__ == '__main__':
    data = load_dataset('/auto/local/data/corpora/Opta/F13M',
                        '/auto/local/data/corpora/Opta/F24')
    data.dump_info()
