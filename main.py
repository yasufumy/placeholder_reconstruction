from argparse import ArgumentParser
import os
import pickle

from chainer.optimizer import GradientClipping, WeightDecay
from chainer import optimizers

from mltools.trainer import Seq2SeqTrainer
from mltools.sampling import Sampling, OrderProvider
from mltools.iterator import SequentialIterator, Iterator
from seq2seq import MLPEncoder2AttentionDecoder, MLPEncoder2GatedAttentionDecoder,\
    DiscriminativeMLPEncoder2AttentionDecoder, DiscriminativeMLPEncoder2GatedAttentionDecoder,\
    DiscriminativeGLUEncoder2GatedAttentionDecoder
from dataset import OptaDataset, EventField, TextField, TextAndContentWordField,\
    TestTextField, TextAndLabelIterator
from utils import Utility, TextFile, compute_class_weight, EndTokenIdRemoval,\
    Seq2SeqWithLabelTrainer, evaluate_bleu_and_accuracy, evaluate_bleu,\
    dump_setting
from config import IGNORE_LABEL


def pickle_dump(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--src-embed', type=int, default=16)
    parser.add_argument('--trg-embed', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--clipping', type=int, default=5)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--event-size', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=20)
    parser.add_argument('--net', type=str, default='plain',
                        choices=('plain', 'gate', 'disc', 'tmpl', 'gate-disc',
                                 'disc-tmpl', 'gate-tmpl', 'gate-disc-tmpl',
                                 'conv-gate-disc-tmpl', 'gate-loss-disc-tmpl'))
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--class-weight', type=float, default=[1., 1.], nargs=2)
    parser.add_argument('--loss-weight', type=float, default=None, nargs=2)
    parser.add_argument('--loss-func', type=str, default='ce',
                        choices=('ce', 'mse'))
    parser.add_argument('--disc-loss', type=int, default=5)
    parser.add_argument('--output', type=str, default='./result')
    parser.add_argument('--dataset', type=str, default='./dataset/parallel.json')
    parser.add_argument('--eval-step', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--numbering', action='store_true', default=False)
    parser.add_argument('--truncate', action='store_true', default=False)
    parser.add_argument('--reverse-decode', action='store_true', default=False)
    parser.add_argument('--bpc', action='store_true', default=False)
    parser.add_argument('--multi-tag', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vocab-size', type=int, default=5500)
    return parser.parse_args()


def training(args):
    source = EventField(fix_length=args.event_size, embed_size=args.src_embed)
    mask_flag = 'tmpl' in args.net
    sentence_size = args.sentence_size if args.truncate else None
    reverse_decode = args.reverse_decode
    if 'disc' in args.net:
        target = TextAndContentWordField(
            start_token=None, fix_length=sentence_size, mask_player=mask_flag,
            mask_team=mask_flag, numbering=args.numbering, reverse=reverse_decode,
            bpc=args.bpc, multi_tag=args.multi_tag)
    else:
        target = TextField(
            start_token=None, fix_length=sentence_size, mask_player=mask_flag,
            mask_team=mask_flag, numbering=args.numbering, reverse=reverse_decode,
            bpc=args.bpc, multi_tag=args.multi_tag)
    if args.truncate:
        train = OptaDataset(
            path=args.dataset + '.train', fields={'source': source, 'target': target})
    else:
        train = OptaDataset(
            path=args.dataset + '.train', fields={'source': source, 'target': target},
            limit_length=args.limit)
    source.build_vocabulary(train.source)
    target.build_vocabulary(train.target, size=args.vocab_size)
    target.player_to_id = source.player_to_id
    target.players = source.id_to_player
    if mask_flag or 'disc' in args.net:
        content_word_to_id = getattr(target, 'content_word_to_id', None)
        target_test = TestTextField(
            source.id_to_player, source.id_to_team, target.word_to_id,
            content_word_to_id, target.unk_id, fix_length=None, bpc=args.bpc)
    else:
        target_test = TextField(
            start_token=None, end_token=None, fix_length=None, bpc=args.bpc)
        target_test.word_to_id = target.word_to_id
        target_test.id_to_word = target.id_to_word
        target_test.unk_id = target.unk_id
    dev = OptaDataset(path=args.dataset + '.dev',
                      fields={'source': source, 'target': target_test}, limit_length=args.limit)
    train2 = OptaDataset(path=args.dataset + '.train',
                         fields={'source': source, 'target': target_test}, limit_length=args.limit)
    test = OptaDataset(path=args.dataset + '.test',
                       fields={'source': source, 'target': target_test})
    test20 = OptaDataset(path=args.dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=20)
    test15 = OptaDataset(path=args.dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=15)
    test10 = OptaDataset(path=args.dataset + '.test',
                         fields={'source': source, 'target': target_test}, limit_length=10)

    start_id, end_id = target.word_to_id['<s>'], target.word_to_id['</s>']
    class_weight = compute_class_weight('./dataset/player_list.txt', target.word_to_id,
                                        args.class_weight[0], args.class_weight[1], gpu=args.gpu)
    dirname = Utility.get_save_directory(
        args.net, './debug' if args.debug else args.output)
    if args.debug:
        save_path = os.path.join('./debug', dirname)
    else:
        save_path = os.path.join(args.output, dirname)
    Utility.make_directory(save_path)
    del args.vocab_size
    setting = {'vocab_size': len(target.word_to_id), 'type_size': len(source.type_to_id),
               'player_size': len(source.player_to_id), 'team_size': len(source.team_to_id),
               'detail_size': len(source.detail_to_id), 'detail_dim': source.details_dimention,
               'start_id': start_id, 'end_id': end_id, 'unk_id': target.unk_id,
               'save_path': save_path, **vars(args)}
    dump_setting(setting, os.path.join(save_path, 'setting.yaml'))
    home_player_tag = target.word_to_id.get(target.home_player_tag)
    away_player_tag = target.word_to_id.get(target.away_player_tag)
    home_team_tag = target.word_to_id.get(target.home_team_tag)
    away_team_tag = target.word_to_id.get(target.away_team_tag)
    print('vocab size: {}'.format(len(target.word_to_id)))
    if args.net == 'plain':
        model = MLPEncoder2AttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed, args.event_size,
            len(target.word_to_id), args.trg_embed, args.hidden, start_id, end_id,
            class_weight, args.mlp_layers, args.max_length, args.dropout, IGNORE_LABEL,
            reverse_decode=reverse_decode)
    elif args.net == 'tmpl':
        model = MLPEncoder2AttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed, args.event_size,
            len(target.word_to_id), args.trg_embed, args.hidden, start_id, end_id,
            class_weight, args.mlp_layers, args.max_length, args.dropout, IGNORE_LABEL,
            source.id_to_player, home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif args.net == 'gate':
        model = MLPEncoder2GatedAttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), args.trg_embed, args.hidden,
            start_id, end_id, class_weight, args.mlp_layers, args.max_length,
            args.dropout, IGNORE_LABEL, reverse_decode=reverse_decode)
    elif args.net == 'gate-tmpl':
        model = MLPEncoder2GatedAttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), args.trg_embed, args.hidden,
            start_id, end_id, class_weight, args.mlp_layers, args.max_length,
            args.dropout, IGNORE_LABEL, source.id_to_player, home_player_tag, away_player_tag,
            source.id_to_team, home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif args.net == 'disc':
        model = DiscriminativeMLPEncoder2AttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), len(target.content_word_to_id),
            args.trg_embed, args.hidden, start_id, end_id, class_weight, args.loss_weight,
            args.disc_loss, args.loss_func, args.mlp_layers, args.max_length, args.dropout,
            IGNORE_LABEL, reverse_decode=reverse_decode)
    elif args.net == 'disc-tmpl':
        model = DiscriminativeMLPEncoder2AttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), len(target.content_word_to_id),
            args.trg_embed, args.hidden, start_id, end_id, class_weight, args.loss_weight,
            args.disc_loss, args.loss_func, args.mlp_layers, args.max_length, args.dropout,
            IGNORE_LABEL, source.id_to_player, home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif args.net == 'gate-disc':
        model = DiscriminativeMLPEncoder2GatedAttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), len(target.content_word_to_id),
            args.trg_embed, args.hidden, start_id, end_id, class_weight, args.loss_weight,
            args.disc_loss, args.loss_func, args.mlp_layers, args.max_length, args.dropout,
            IGNORE_LABEL, reverse_decode=reverse_decode)
    elif args.net == 'gate-disc-tmpl':
        model = DiscriminativeMLPEncoder2GatedAttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), len(target.content_word_to_id),
            args.trg_embed, args.hidden, start_id, end_id, class_weight, args.loss_weight,
            args.disc_loss, args.loss_func, args.mlp_layers, args.max_length, args.dropout,
            IGNORE_LABEL, source.id_to_player, home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)
    elif args.net == 'conv-gate-disc-tmpl':
        model = DiscriminativeGLUEncoder2GatedAttentionDecoder(
            len(source.type_to_id), len(source.player_to_id), len(source.team_to_id),
            len(source.detail_to_id), source.details_dimention, args.src_embed,
            args.event_size, len(target.word_to_id), len(target.content_word_to_id),
            args.trg_embed, args.hidden, start_id, end_id, class_weight, args.loss_weight,
            args.disc_loss, args.loss_func, args.mlp_layers, args.max_length, args.dropout,
            IGNORE_LABEL, source.id_to_player, home_player_tag, away_player_tag, source.id_to_team,
            home_team_tag, away_team_tag, target.player_to_id, target.players,
            reverse_decode=reverse_decode)

    model.keyword_ids = [target.word_to_id['save'], target.word_to_id['block'],
                         target.word_to_id['chance'], target.word_to_id['shot'],
                         target.word_to_id['clearance'], target.word_to_id['kick'],
                         target.word_to_id['ball'], target.word_to_id['blocked'],
                         target.word_to_id['denied']]
    model.id_to_word = target.id_to_word
    if args.numbering:
        model.player_id = target.player_id
        model.team_id = target.team_id

    if args.gpu is not None:
        model.use_gpu(args.gpu)
    opt = optimizers.Adam(args.lr)
    opt.setup(model)
    if args.clipping > 0:
        opt.add_hook(GradientClipping(args.clipping))
    if args.decay > 0:
        opt.add_hook(WeightDecay(args.decay))

    N = len(train.source)
    batch_size = args.batch
    order_provider = OrderProvider(Sampling.get_random_order(N))
    src_train_iter = SequentialIterator(train.source, batch_size, order_provider,
                                        args.event_size, source.fillvalue, gpu=args.gpu)
    if 'disc' in args.net:
        trg_train_iter = TextAndLabelIterator(train.target, batch_size, order_provider,
                                              args.sentence_size, IGNORE_LABEL, gpu=args.gpu)
    else:
        trg_train_iter = SequentialIterator(train.target, batch_size, order_provider,
                                            args.sentence_size, IGNORE_LABEL, gpu=args.gpu)
    src_dev_iter = SequentialIterator(dev.source, batch_size, None, args.event_size,
                                      source.fillvalue, gpu=args.gpu)
    trg_dev_iter = Iterator(dev.target, batch_size,
                            wrapper=EndTokenIdRemoval(end_id), gpu=None)
    src_test_iter = SequentialIterator(test.source, batch_size, None, args.event_size,
                                       source.fillvalue, gpu=args.gpu)
    src_test20_iter = SequentialIterator(test20.source, batch_size, None, args.event_size,
                                         source.fillvalue, gpu=args.gpu)
    src_test15_iter = SequentialIterator(test15.source, batch_size, None, args.event_size,
                                         source.fillvalue, gpu=args.gpu)
    src_test10_iter = SequentialIterator(test10.source, batch_size, None, args.event_size,
                                         source.fillvalue, gpu=args.gpu)
    src_train2_iter = SequentialIterator(train2.source, batch_size, None, args.event_size,
                                         source.fillvalue, gpu=args.gpu)
    trg_train2_iter = Iterator(train2.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test_iter = Iterator(test.target, batch_size,
                             wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test20_iter = Iterator(test20.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test15_iter = Iterator(test15.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    trg_test10_iter = Iterator(test10.target, batch_size,
                               wrapper=EndTokenIdRemoval(end_id), gpu=None)
    if 'disc' in args.net:
        trainer = Seq2SeqWithLabelTrainer(model, opt, src_train_iter, trg_train_iter,
                                          src_dev_iter, trg_dev_iter, order_provider,
                                          evaluate_bleu_and_accuracy, args.epoch, save_path,
                                          args.eval_step, src_train2_iter, trg_train2_iter)
    else:
        trainer = Seq2SeqTrainer(model, opt, src_train_iter, trg_train_iter,
                                 src_dev_iter, trg_dev_iter, order_provider, evaluate_bleu,
                                 args.epoch, save_path, args.eval_step,
                                 src_train2_iter, trg_train2_iter)

    trainer.run()

    # load best model
    model.load_model(os.path.join(save_path, 'best.model'))
    if 'disc' in args.net:
        bleu_score_dev, _, _ = evaluate_bleu_and_accuracy(
            model, src_dev_iter, trg_dev_iter)
        bleu_score, _, _ = evaluate_bleu_and_accuracy(
            model, src_test_iter, trg_test_iter)
        bleu_score20, _, hypotheses = evaluate_bleu_and_accuracy(
            model, src_test20_iter, trg_test20_iter)
        bleu_score15, _, _ = evaluate_bleu_and_accuracy(
            model, src_test15_iter, trg_test15_iter)
        bleu_score10, _, _ = evaluate_bleu_and_accuracy(
            model, src_test10_iter, trg_test10_iter)
    else:
        bleu_score_dev, _ = evaluate_bleu(
            model, src_dev_iter, trg_dev_iter)
        bleu_score, _ = evaluate_bleu(
            model, src_test_iter, trg_test_iter)
        bleu_score20, hypotheses = evaluate_bleu(
            model, src_test20_iter, trg_test20_iter)
        bleu_score15, _ = evaluate_bleu(
            model, src_test15_iter, trg_test15_iter)
        bleu_score10, _ = evaluate_bleu(
            model, src_test10_iter, trg_test10_iter)
    TextFile(os.path.join(save_path, 'hypotheses.txt'),
             [' '.join(ys) for ys in trainer.hypotheses]).save()
    print('dev score: {}'.format(bleu_score_dev))
    print('test score: {}'.format(bleu_score))
    print('test score20: {}'.format(bleu_score20))
    print('test score15: {}'.format(bleu_score15))
    print('test score10: {}'.format(bleu_score10))

    # saving fields
    pickle_dump(os.path.join(save_path, 'source.pkl'), source)
    pickle_dump(os.path.join(save_path, 'target.pkl'), target)
    pickle_dump(os.path.join(save_path, 'target_test.pkl'), target_test)


if __name__ == '__main__':
    args = parse_args()
    training(args)
