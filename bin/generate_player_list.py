#!/usr/bin/env python

import os
import json
import xml.etree.ElementTree as ET
from glob import glob
from argparse import ArgumentParser
from unidecode import unidecode


def to_ascii_str(string):
    return unidecode(string).encode('ascii').decode('utf-8')


class PlayerListGenerator:
    def __init__(self, data_path='/auto/local/data/corpora/Opta/F9',
                 save_path='../dataset/player_list.txt'):
        if not os.path.isdir(data_path):
            raise NotADirectoryError('{} is not found.'.format(data_path))
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            raise NotADirectoryError('{} is not a directory'.format(save_dir))

        self.data_path = data_path
        self.save_path = save_path

    def run(self):
        players = {}
        for path in glob(self.data_path + '/*.xml'):
            root = ET.parse(path).getroot()
            for p in root.findall('.//Player'):
                first_name = to_ascii_str(p.find('.//First').text.lower())
                last_name = to_ascii_str(p.find('.//Last').text.lower())
                player_id = int(p.get('uID')[1:])
                players[player_id] = '{} {}'.format(first_name, last_name)

        with open(self.save_path, 'w', encoding='utf-8') as fp:
            json.dump(players, fp, indent=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/auto/local/data/corpora/Opta/F9')
    parser.add_argument('--save-path', type=str, default='../dataset/player_list.text')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    PlayerListGenerator(args.data_path, args.save_path).run()
