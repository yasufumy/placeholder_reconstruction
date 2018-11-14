#!/usr/bin/env python

import os
import json
import xml.etree.ElementTree as ET
from glob import glob
from argparse import ArgumentParser
from unidecode import unidecode


def to_ascii_str(string):
    return unidecode(string).encode('ascii').decode('utf-8')


class TeamListGenerator:
    def __init__(self, data_path='/auto/local/data/corpora/Opta/F9',
                 save_path='../dataset/team_list.json'):
        if not os.path.isdir(data_path):
            raise NotADirectoryError('{} is not found.'.format(data_path))
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            raise NotADirectoryError('{} is not a directory'.format(save_dir))

        self.data_path = data_path
        self.save_path = save_path

    def run(self):
        teams = {}
        for path in glob(self.data_path + '/*.xml'):
            root = ET.parse(path).getroot()
            for t in root.findall('.//Team'):
                team_id = int(t.get('uID')[1:])
                teams[team_id] = t.find('./Name').text

        with open(self.save_path, 'w', encoding='utf-8') as fp:
            json.dump(teams, fp, indent=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/auto/local/data/corpora/Opta/F9')
    parser.add_argument('--save-path', type=str, default='../dataset/team_list.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    TeamListGenerator(args.data_path, args.save_path).run()
