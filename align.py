#!/usr/bin/env python

import os
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import re

from config import event_type_mapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--game-id', type=str, default='./dataset/gameid.txt')
    parser.add_argument('--data-path', type=str, default='/auto/local/data/corpora/Opta')
    parser.add_argument('--dest-path', type=str, default='./dataset/algin.txt')
    parser.add_argument('--delta', type=int, default=5)
    return parser.parse_args()


def main(args):
    fp = open(args.game_id)
    basepath = args.data_path
    save_file = open(args.dest_path, 'w')
    ignore_types = ('full time', 'half time', 'kick off', 'pre kick off',
                    'second half')

    for game_id in fp:
        game_id = game_id.rstrip()
        path_f24 = '{}/F24/F24_gameid{}.xml'.format(basepath, game_id)
        if not os.path.exists(path_f24):
            print('File is not found on {}'.format(path_f24))
            return

        path_f9 = '{}/F9/F9_gameid{}.xml'.format(basepath, game_id)
        if not os.path.exists(path_f9):
            print('File is not found on {}'.format(path_f9))
            return

        path_f13m = '{}/F13M/F13M_gameid{}.xml'.format(basepath, game_id)
        if not os.path.exists(path_f13m):
            print('File is not found on {}'.format(path_f13m))
            return

        id_to_player = {}
        tree_f9 = ET.parse(path_f9).getroot()
        for t in tree_f9.findall('.//Team'):
            team_id = int(t.get('uID').replace('t', ''))
            id_to_player[team_id] = {}
            for p in t.findall("./Player"):
                player_id = int(p.get('uID').replace('p', ''))
                first_name = p.find('.//First').text
                last_name = p.find('.//Last').text
                id_to_player[player_id] = '({}|{})'.format(first_name, last_name)

        tree = ET.parse(path_f24).getroot()

        tree_comment = ET.parse(path_f13m).getroot()
        for c in list(tree_comment):
            if c.get('type') in ignore_types:
                continue
            minute = int(c.get('minute'))
            cm = c.get('comment')
            candidates = []
            for delta in range(- args.delta, args.delta + 1):
                elements = tree.findall('.//Event[@min="{}"]'.format(minute + delta))
                for e in elements:
                    tmp = e.get('player_id')
                    player_id = int(tmp) if tmp else tmp
                    if player_id is not None:
                        if re.search(id_to_player[player_id], cm):
                            candidate = event_type_mapper[int(e.get('type_id'))]. \
                                replace(" ", "_") + ":" + e.get('id')
                            candidates.append(candidate)

            if candidates:
                print("{}\t{}\t{}\t{}".format(cm, " ".join(candidates), game_id, c.get('id')),
                      file=save_file, flush=True)
    save_file.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
