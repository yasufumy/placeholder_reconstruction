from argparse import ArgumentParser
import os
import json
import pickle

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from mltools.iterator import SequentialIterator, Iterator
from seq2seq import MLPEncoder2AttentionDecoder, MLPEncoder2GatedAttentionDecoder,\
    DiscriminativeMLPEncoder2AttentionDecoder, DiscriminativeMLPEncoder2GatedAttentionDecoder
from dataset import OptaDataset
from utils import TextFile, EndTokenIdRemoval,\
    evaluate_bleu_and_accuracy, evaluate_bleu, load_setting
from config import IGNORE_LABEL, event_type_mapper, qualifier_type_mapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    return parser.parse_args()


def pickle_load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def evaluation(args):
    source = pickle_load(os.path.join(args.model_path, 'source.pkl'))
    target = pickle_load(os.path.join(args.model_path, 'target.pkl'))
    target_test = pickle_load(os.path.join(args.model_path, 'target_test.pkl'))
    setting = load_setting(os.path.join(args.model_path, 'setting.yaml'))
    start_id, end_id = setting['start_id'], setting['end_id']
    type_size = setting['type_size']
    player_size = setting['player_size']
    team_size = setting['team_size']
    detail_size = setting['detail_size']
    detail_dim = setting['detail_dim']
    src_embed = setting['src_embed']
    event_size = setting['event_size']
    vocab_size = setting['vocab_size']
    trg_embed = setting['trg_embed']
    hidden = setting['hidden']
    start_id = setting['start_id']
    end_id = setting['end_id']
    class_weight = None
    mlp_layers = setting['mlp_layers']
    max_length = setting['max_length']
    dropout = setting['dropout']
    loss_weight = None
    disc_loss = setting['disc_loss']
    loss_func = setting['loss_func']
    net = setting['net']
    dataset = setting['dataset']
    numbering = setting['numbering']
    reverse_decode = setting['reverse_decode']
    home_player_tag = target.word_to_id.get(target.home_player_tag)
    away_player_tag = target.word_to_id.get(target.away_player_tag)
    home_team_tag = target.word_to_id.get(target.home_team_tag)
    away_team_tag = target.word_to_id.get(target.away_team_tag)
    test = OptaDataset(path=dataset + '.test',
                       fields={'source': source, 'target': target_test})
    test20 = OptaDataset(path=dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=20)
    test15 = OptaDataset(path=dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=15)
    test10 = OptaDataset(path=dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=10)

    if 'disc' in net:
        content_word_size = len(target.content_word_to_id)
    print('vocab size: {}'.format(vocab_size))
    if net == 'plain':
        model = MLPEncoder2AttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, trg_embed, hidden, start_id, end_id,
            class_weight, mlp_layers, max_length, dropout, IGNORE_LABEL,
            reverse_decode=reverse_decode)
    elif net == 'tmpl':
        model = MLPEncoder2AttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, trg_embed, hidden, start_id, end_id,
            class_weight, mlp_layers, max_length, dropout, IGNORE_LABEL,
            source.id_to_player, home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif net == 'gate':
        model = MLPEncoder2GatedAttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, trg_embed, hidden, start_id, end_id,
            class_weight, mlp_layers, max_length, dropout, IGNORE_LABEL,
            reverse_decode=reverse_decode)
    elif net == 'gate-tmpl':
        model = MLPEncoder2GatedAttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, trg_embed, hidden, start_id, end_id,
            class_weight, mlp_layers, max_length, dropout, IGNORE_LABEL, source.id_to_player,
            home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif net == 'disc':
        model = DiscriminativeMLPEncoder2AttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, content_word_size, trg_embed, hidden,
            start_id, end_id, class_weight, loss_weight,
            disc_loss, loss_func, mlp_layers, max_length, dropout, IGNORE_LABEL,
            reverse_decode=reverse_decode)
    elif net == 'disc-tmpl':
        model = DiscriminativeMLPEncoder2AttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, content_word_size, trg_embed, hidden,
            start_id, end_id, class_weight, loss_weight, disc_loss, loss_func,
            mlp_layers, max_length, dropout, IGNORE_LABEL, source.id_to_player,
            home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif net == 'gate-disc':
        model = DiscriminativeMLPEncoder2GatedAttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, content_word_size, trg_embed, hidden,
            start_id, end_id, class_weight, loss_weight,
            disc_loss, loss_func, mlp_layers, max_length, dropout, IGNORE_LABEL,
            reverse_decode=reverse_decode)
    elif net == 'gate-disc-tmpl':
        model = DiscriminativeMLPEncoder2GatedAttentionDecoder(
            type_size, player_size, team_size, detail_size, detail_dim, src_embed,
            event_size, vocab_size, content_word_size, trg_embed, hidden,
            start_id, end_id, class_weight, loss_weight, disc_loss, loss_func,
            mlp_layers, max_length, dropout, IGNORE_LABEL, source.id_to_player,
            home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    if numbering:
        model.player_id = target.player_id
        model.team_id = target.team_id
    # load best model
    if args.gpu is not None:
        model.use_gpu(args.gpu)
    model.id_to_word = target.id_to_word
    model.load_model(os.path.join(args.model_path, 'best.model'))
    batch_size = args.batch
    src_test_iter = SequentialIterator(test.source, batch_size, None, event_size,
                                       source.fillvalue, gpu=args.gpu)
    src_test20_iter = SequentialIterator(test20.source, batch_size, None, event_size,
                                         source.fillvalue, gpu=args.gpu)
    src_test15_iter = SequentialIterator(test15.source, batch_size, None, event_size,
                                         source.fillvalue, gpu=args.gpu)
    src_test10_iter = SequentialIterator(test10.source, batch_size, None, event_size,
                                         source.fillvalue, gpu=args.gpu)
    trg_test_iter = Iterator(test.target, batch_size,
                             wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test20_iter = Iterator(test20.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test15_iter = Iterator(test15.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test10_iter = Iterator(test10.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)

    with open('./dataset/player_list.json.new') as f:
        id_to_player = json.load(f)
    with open('./dataset/team_list.json.new') as f:
        id_to_team = json.load(f)

    def convert(ind, no_tag=False):
        if 'player' in ind:
            if no_tag:
                i = ind.replace('player', '')
                return id_to_player.get(i, ind)
            else:
                return ind
        elif 'team' in ind:
            if no_tag:
                i = ind.replace('team', '')
                return id_to_team.get(i, ind)
            else:
                return ind
        else:
            return ind
    if 'disc' in net:
        bleu_score, accuracy, hypotheses = evaluate_bleu_and_accuracy(
            model, src_test_iter, trg_test_iter)
        bleu_score20, _, hypotheses20 = evaluate_bleu_and_accuracy(
            model, src_test20_iter, trg_test20_iter)
        bleu_score15, _, hypotheses15 = evaluate_bleu_and_accuracy(
            model, src_test15_iter, trg_test15_iter)
        bleu_score10, _, hypotheses10 = evaluate_bleu_and_accuracy(
            model, src_test10_iter, trg_test10_iter)
    else:
        bleu_score, hypotheses = evaluate_bleu(
            model, src_test_iter, trg_test_iter)
        bleu_score20, hypotheses20 = evaluate_bleu(
            model, src_test20_iter, trg_test20_iter)
        bleu_score15, hypotheses15 = evaluate_bleu(
            model, src_test15_iter, trg_test15_iter)
        bleu_score10, hypotheses10 = evaluate_bleu(
            model, src_test10_iter, trg_test10_iter)

    print('best score: {}'.format(bleu_score))
    print('best score20: {}'.format(bleu_score20))
    print('best score15: {}'.format(bleu_score15))
    print('best score10: {}'.format(bleu_score10))
    # save hypothesis
    hypotheses_for_save = [' '.join([convert(y, True) for y in h]) for h in hypotheses]
    hypotheses20_for_save = [' '.join([convert(y, True) for y in h]) for h in hypotheses20]
    hypotheses15_for_save = [' '.join([convert(y, True) for y in h]) for h in hypotheses15]
    hypotheses10_for_save = [' '.join([convert(y, True) for y in h]) for h in hypotheses10]
    references_for_save = [' '.join(convert(y, True) for y in r[0]) for r in test.target]
    references20_for_save = [' '.join(convert(y, True) for y in r[0]) for r in test20.target]
    references15_for_save = [' '.join(convert(y, True) for y in r[0]) for r in test15.target]
    references10_for_save = [' '.join(convert(y, True) for y in r[0]) for r in test10.target]
    TextFile(os.path.join(args.model_path, 'hypo'), hypotheses_for_save).save()
    TextFile(os.path.join(args.model_path, 'hypo_len20'), hypotheses20_for_save).save()
    TextFile(os.path.join(args.model_path, 'hypo_len15'), hypotheses15_for_save).save()
    TextFile(os.path.join(args.model_path, 'hypo_len10'), hypotheses10_for_save).save()
    TextFile(os.path.join('./dataset', 'ref'), references_for_save).save()
    TextFile(os.path.join('./dataset', 'ref_len20'), references20_for_save).save()
    TextFile(os.path.join('./dataset', 'ref_len15'), references15_for_save).save()
    TextFile(os.path.join('./dataset', 'ref_len10'), references10_for_save).save()
    # generate readable text
    result = []
    for ref, hyp in zip(test.target.data, hypotheses):
        if type(ref) == tuple:
            ref = ref[0]
        ref = ' '.join([convert(y) for y in ref]).split()
        try:
            bleu_score = sentence_bleu(
                [ref], hyp, smoothing_function=SmoothingFunction().method1)
        except:
            bleu_score = 0
        ref = ' '.join([convert(y, True) for y in ref]).split()
        hyp = ' '.join([convert(y, True) for y in hyp]).split()
        result.append((' '.join(ref), ' '.join(hyp), bleu_score))
    inputs = []
    for xs in test20.source.data:
        data = []
        for x in xs[:5]:
            event = event_type_mapper.get(x[0], x[0])
            player = id_to_player.get(str(x[1]), x[1])
            team = id_to_team.get(str(x[2]), x[2])
            detail = ','.join(
                [qualifier_type_mapper.get(i[-1], i[-1]) for i in x[-1]])
            data.append('event: {} player: {} team: {} detail: {}'.format(
                event, player, team, detail))
        inputs.append('\n'.join(data))
    result = [[x, *y] for x, y in zip(inputs, result)]
    result = sorted(result, key=lambda x: -x[-1])
    TextFile(
        os.path.join(args.model_path, 'test20_gate_disc_tmpl.txt'),
        ['src:\n{}\nref: {}\nhyp: {}\nbleu: {}\n##\n'.format(*x) for x in result]).save()


if __name__ == '__main__':
    args = parse_args()
    evaluation(args)
