import json
from argparse import ArgumentParser

from dataset import TextAndContentWordField
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


with open('./dataset/player_list.json') as f:
    players = json.load(f)
with open('./dataset/team_list.json') as f:
    teams = json.load(f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--hypo-path', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--event-length', type=int)
    parser.add_argument('--sentence-length', type=int)
    return parser.parse_args()


def replace(word):
    if target_field.player_pat.match(word) is not None:
        return players[word.replace('player', '')]
    elif target_field.team_pat.match(word) is not None:
        return teams[word.replace('team', '')]
    else:
        return word


def convert(target):
    text = target_field._tokenize(target)
    text = target_field._mask(target, text, True)
    return [replace(w) for w in text[:target_field.fix_length.stop]]


def output_result(json_file, hypo_file, save_file):
    global target_field
    target_field = TextAndContentWordField(None, None, args.sentence_length,
                                           mask_player=True, mask_team=True)
    with open(json_file) as f:
        data = json.load(f)
    with open(hypo_file) as f:
        hypo = [line.replace('\n', '') for line in f]
    result = []
    for x, hypo in zip(data, hypo):
        result.append({'src': 'src:\n{}'.format('\n'.join(
                       [v['desc'] for i, v in enumerate(x['source']) if i < args.event_length])),
                       'trg': convert(x['target']), 'hyp': hypo.split()})

    def key_func(x):
        return - sentence_bleu([x['trg']], x['hyp'],
                               smoothing_function=SmoothingFunction().method1)
    result = sorted(result, key=key_func)
    fp = open(save_file, 'w')
    for x in result:
        print(x['src'], file=fp, flush=True)
        print('trg: {}'.format(' '.join(x['trg'])), file=fp, flush=True)
        print('hyp: {}'.format(' '.join(x['hyp'])), file=fp, flush=True)
        print('#####\n', file=fp, flush=True)
    fp.close()


def main(args):
    output_result(args.data_path, args.hypo_path, args.save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
