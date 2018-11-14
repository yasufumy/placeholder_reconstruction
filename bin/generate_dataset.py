#!/usr/bin/env python

import os
import re
import sys
import json
import copy
import random
import pickle
from math import ceil
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from unidecode import unidecode

from nltk.tokenize import sent_tokenize

sys.path.append('.')
from train_em import EMAlgorism, FeaturedEM  # noqa: F401
from config import event_type_mapper, qualifier_type_mapper  # noqa: E402


with open('./dataset/player_list.json.new') as f:
    player_id_to_name = json.load(f)

with open('./dataset/team_list.json.new') as f:
    team_id_to_name = json.load(f)


def to_ascii_str(string):
    return unidecode(string).encode('ascii').decode('utf-8')


class DataGenerator:
    def __init__(self, data_path='/auto/local/data/corpora/Opta',
                 game_id_file_path='../dataset/gameid.txt',
                 save_path='../dataset/parallel.json', time_range=5, split=True,
                 player_align=False, seed=123):
        if not os.path.isfile(game_id_file_path):
            raise FileNotFoundError('{} is not found'.format(game_id_file_path))
        if not os.path.isdir(data_path):
            raise NotADirectoryError('{} is not a directory'.format(data_path))
        if not os.path.isdir(os.path.dirname(save_path)):
            raise NotADirectoryError('{} is not a directory'.format(save_path))
        self.data_path = data_path
        self.save_path = save_path
        self.game_id_file_path = game_id_file_path
        self.time_range = time_range
        self.seed = seed
        self.split = split
        self.player_align = player_align
        self.ignore_types = ('full time', 'half time', 'kick off', 'pre kick off',
                             'second half')
        self.ignore_periods = ('0', '14', '15', '16')
        self.ignore_qids = {'213', '56', '140', '141', '212'}
        self.ignore_eids = {'25', '27', '28', '30', '32', '34', '37', '43', '71', '75'}
        self.skip_qids = {102, 103, 146, 147, 55, 216, 233, 56, 140, 141,
                          212, 213, 230, 231}

    def run(self):
        with open(self.game_id_file_path) as fp:
            game_ids = [line.strip() for line in fp]
        # shuffle data
        random.seed(self.seed)
        random.shuffle(game_ids)
        # split data
        data_length = len(game_ids)
        if self.split:
            train_size = ceil(data_length * 0.8)
            test_size = train_size + ceil(data_length * 0.1)
            ext = '.train'
        else:
            train_size = data_length - 1
            test_size = -1
            ext = '.all'

        data_path = self.data_path
        dataset = []

        def _get_target_path(target_type, game_id):
            path = os.path.join(data_path,
                                '{}/{}_gameid{}.xml'.format(target_type, target_type, game_id))
            if not os.path.isfile(path):
                raise FileNotFoundError('{} is not found'.format(path))
            return path

        def _compute_player_dict(f9_path):
            id_to_player = {}
            f9_root = ET.parse(f9_path).getroot()
            for t in f9_root.findall('.//Team'):
                team_id = t.get('uID').replace('t', '')
                id_to_player[team_id] = {}
                for p in t.findall('./Player'):
                    player_id = p.get('uID').replace('p', '')
                    first_name = p.find('.//First').text
                    last_name = p.find('.//Last').text
                    id_to_player[team_id][player_id] = '({}|{}|{}|{})'.format(
                        first_name, last_name, to_ascii_str(first_name), to_ascii_str(last_name))
            return id_to_player

        for i, game_id in enumerate(game_ids):
            game_id = game_id.strip()
            f9_path = _get_target_path('F9', game_id)
            f13m_path = _get_target_path('F13M', game_id)
            f24_path = _get_target_path('F24', game_id)

            id_to_player = _compute_player_dict(f9_path)
            data = self._align_data(f13m_path, f24_path, id_to_player)
            dataset.extend(data)
            if i == train_size:
                with open(self.save_path + ext, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2)
                dataset = []
            elif i == test_size:
                with open(self.save_path + '.test', 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2)
                dataset = []

        if dataset:
            with open(self.save_path + '.dev', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)

    def _align_data(self, f13m_path, f24_path, id_to_player):
        f13m_root = ET.parse(f13m_path).getroot()
        messages = f13m_root.findall('.//message')
        home_team_name = f13m_root.get('home_team_name')
        away_team_name = f13m_root.get('away_team_name')
        f24_root = ET.parse(f24_path).getroot()
        game = f24_root.find('./Game')
        home_team_id = game.get('home_team_id')
        away_team_id = game.get('away_team_id')
        data = []
        for message in messages:
            if message.get('type') in self.ignore_types:
                continue
            if message.get('period') in self.ignore_periods:
                continue
            comment = message.get('comment')
            if not comment or comment == '\xa0':
                continue
            comments = sent_tokenize(comment)
            for comment in comments:
                message = copy.deepcopy(message)
                message.set('comment', comment)
                minute = int(message.get('minute'))
                comment_id = message.get('id')
                home_players = {}
                away_players = {}
                candidates = []
                for delta in sorted(range(- self.time_range, self.time_range + 1),
                                    key=lambda x: abs(x)):
                    events = f24_root.findall('.//Event[@min="{}"]'.format(minute + delta))
                    for event in events:
                        player_id = event.get('player_id')
                        if player_id is None:
                            continue
                        qids = {q.get('qualifier_id') for q in event.findall('./Q')}
                        if qids == self.ignore_qids and \
                           event.get('type_id') == '1' and not event.get('keypass'):
                            continue
                        if event.get('type_id') in self.ignore_eids:
                            continue
                        team_id = event.get('team_id')
                        player = id_to_player[team_id][player_id]
                        if re.search(player, comment):
                            if team_id == home_team_id:
                                home_players[player] = player_id
                            elif team_id == away_team_id:
                                away_players[player] = player_id
                            candidates.append(self._convert(event, minute))
                            # 55, 53, 7
                            addtional_events = f24_root.findall(
                                './/Event[@min="{}"]/Q[@value="{}"]/..'.format(
                                    minute + delta, player_id))
                            if addtional_events:
                                addtional_events = [
                                    self._convert(x, minute) for x in addtional_events
                                    if x.get('player_id')]
                                candidates.extend(addtional_events)

                        # elif self.player_align:
                        #     candidates.append(self._convert(event, minute))
                    # if candidates:
                    #     break

                if candidates:
                    data.append({'target': {'comment': comment, 'id': comment_id,
                                            'home_players': home_players,
                                            'away_players': away_players,
                                            'home_team_name': home_team_name,
                                            'away_team_name': away_team_name,
                                            'home_team_id': home_team_id,
                                            'away_team_id': away_team_id},
                                 'source': candidates})
                    break
        return data

    def _convert(self, event, comment_minute):
        id_ = int(event.get('id'))
        event_id = int(event.get('event_id'))
        type_id = int(event.get('type_id'))
        team_id = int(event.get('team_id'))
        player_id = int(event.get('player_id'))
        minute = float(event.get('min'))
        second = float(event.get('sec'))
        delta = (comment_minute - minute) * 60 - second
        outcome = int(event.get('outcome'))
        x = float(event.get('x')) / 100
        y = float(event.get('y')) / 100
        qs = [(int(q.attrib['qualifier_id']), q.attrib.get('value'))
              for q in event.findall('./Q')]
        type_outcome_q = [(type_id, outcome, q[0]) for q in qs if q[0] not in self.skip_qids]
        end_xy = {'x' if q[0] == 140 else 'y': float(q[1]) / 100
                  for q in qs if q[0] in (140, 141)}
        desc = 'name: {}, team: {}, event: {}, detail: {}'.format(
            player_id_to_name[str(player_id)], team_id_to_name[str(team_id)],
            event_type_mapper[type_id], ','.join([qualifier_type_mapper[q[0]] for q in qs])
        )
        return {'event_id': event_id, 'type_id': type_id, 'team_id': team_id,
                'player_id': player_id, 'minute': minute, 'second': second,
                'outcome': outcome, 'x': x, 'y': y, 'end_x': end_xy.get('x', -1),
                'end_y': end_xy.get('y', -1), 'details': type_outcome_q,
                'id': id_, 'delta': delta, 'desc': desc}


class EMDataGenerator(DataGenerator):
    def __init__(self, data_path='/auto/local/data/corpora/Opta',
                 game_id_file_path='../dataset/gameid.txt',
                 save_path='../dataset/parallel.json', time_range=5, seed=123,
                 em_path='./result/em.pickle'):
        super().__init__(data_path, game_id_file_path, save_path, time_range,
                         True, seed)
        self.em_path = em_path

    def run(self):
        with open(self.em_path, 'rb') as fp:
            self.em = pickle.load(fp)
        super().run()

    def _align_data(self, f13m_path, f24_path, id_to_player):
        messages = ET.parse(f13m_path).getroot().findall('.//message')
        f24_root = ET.parse(f24_path).getroot()
        data = []
        for message in messages:
            if message.get('type') in self.ignore_types:
                continue
            minute = int(message.get('minute'))
            comment = message.get('comment')
            comment_id = message.get('id')
            players = []
            for delta in sorted(range(- self.time_range, self.time_range + 1),
                                key=lambda x: abs(x)):
                events = f24_root.findall('.//Event[@min="{}"]'.format(minute + delta))
                for event in events:
                    player_id = event.get('player_id')
                    if player_id is None:
                        continue
                    team_id = event.get('team_id')
                    if re.search(id_to_player[team_id][player_id], comment):
                        players.append(id_to_player[team_id][player_id])
            event_ids = self.em.compute_events(comment_id)
            if event_ids is None:
                continue
            events = []
            for event_id in event_ids:
                event = f24_root.find('.//Event[@id="{}"]'.format(event_id))
                events.append(self._convert(event, minute))
            data.append({'target': {'comment': comment, 'id': comment_id,
                                    'players': list(set(players))},
                         'source': events})
        return data


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/auto/local/data/corpora/Opta')
    parser.add_argument('--game-id-file', type=str, default='./dataset/gameid.txt')
    parser.add_argument('--save-path', type=str, default='./dataset/parallel.json')
    parser.add_argument('--em-path', type=str, default=None)
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--player-align', action='store_true', default=False)
    parser.add_argument('--time-range', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.em_path is None:
        DataGenerator(args.data_path, args.game_id_file, args.save_path,
                      args.time_range, args.split, args.player_align, args.seed).run()
    else:
        EMDataGenerator(args.data_path, args.game_id_file, args.save_path,
                        args.time_range, args.seed, args.em_path).run()
