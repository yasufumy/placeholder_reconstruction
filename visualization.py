#!/usr/bin/env python

import platform
import os
import sys
from argparse import ArgumentParser
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt

from dataset import load_players

if platform.system() != 'Darwin':
    exit('Execute this script on Mac.')

matplotlib.use('TkAgg', warn=False)
plt.switch_backend('tkagg')

pitch_width = 110
pitch_height = 75


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--xml-path', type=str,
                        default='/auto/local/data/corpora/Opta/F24/F24_gameid803453.xml')
    parser.add_argument('--event-id', type=int, default=None)
    parser.add_argument('--f42-path', type=str,
                        default='/auto/local/data/corpora/Opta/F42/F42_competiiton8_seasonid2015.xml')
    return parser.parse_args()


def parse_event(event_element, id_to_player):
    x, y = float(event_element.attrib['x']), float(event_element.attrib['y'])
    qs = event_element.findall('./Q')
    end_x = end_y = -1
    for q in qs:
        if q.attrib['qualifier_id'] == '140':
            end_x = float(q.attrib['value'])
        elif q.attrib['qualifier_id'] == '141':
            end_y = float(q.attrib['value'])
    return x, y, end_x, end_y, id_to_player[int(event_element.attrib['player_id'])]

def draw_event(x, y, end_x, end_y, player_name):
    x = pitch_width * x / 100
    y = pitch_height * y / 100
    if end_x > 0 and end_y > 0:
        end_x = pitch_width * end_x / 100
        end_y = pitch_height * end_y / 100
        # ball
        plt.quiver(x, y, end_x - x, end_y - y, color='k', angles='xy', scale_units='xy', scale=1)
        plt.plot(x, y, 'o', color='k')
        plt.plot(end_x, end_y, 'o', color='k')
    # player
    plt.plot(x, y, '+', color='b')
    plt.annotate(player_name, xy=(x, y), size=10)
    plt.xlim([0, pitch_width])
    plt.ylim([0, pitch_height])


def draw_pitch():
    center_x = pitch_width / 2
    center_y = pitch_height / 2
    plt.plot([center_x, center_x], [0, pitch_height], color='k')
    circle = plt.Circle([center_x, center_y], radius=9.15, color='k', fill=False)
    # referred https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    plt.gcf().gca().add_artist(circle)


def main(args):
    if not os.path.exists(args.xml_path):
        print('File is not found on {}'.format(args.xml_path))
        return
    tree = ET.parse(args.xml_path).getroot()
    team_id = load_players(args.f42_path)
    id_to_player = {i: p[0] for k, v in team_id.items() for p, i in v.items()}

    if args.event_id is None:
        plt.ion()
        for x in tree.findall('.//Event'):
            if x.attrib['type_id'] != '1':
                continue
            plt.clf()
            draw_pitch()
            draw_event(*parse_event(x, id_to_player))
            plt.pause(0.1)
    else:
        element = tree.find('.//Event[@id="{}"]'.format(args.event_id))

        if element is None:
            print('No event found (assgined id: {})'.format(args.event_id))
            return
        plt.clf()
        draw_pitch()
        draw_event(*parse_event(element, id_to_player))
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
