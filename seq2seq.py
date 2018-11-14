import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from itertools import takewhile
from functools import reduce

from mltools.model import BaseModel

from encoders import MLPEncoder, GLUEncoder
from decoders import LSTMDecoder, AttentionDecoder
from functions import weighted_sigmoid_cross_entropy


def transpose(array):
    return np.transpose(array).tolist()


def remove_eos(array, end_token_id):
    result = []
    for x in array:
        temp = list(takewhile(lambda i: i != end_token_id, x))
        result.append(temp or x[0])
    return result


class MLPEncoder2LSTMDecoder(BaseModel):
    def __init__(self, type_size, player_size, team_size, detail_size, detail_dim,
                 src_embed_size, sequence_length, vocab_size, trg_embed_size,
                 hidden_size, start_token_id, end_token_id, mlp_layers=2,
                 max_length=20, dropout_ratio=0.2, ignore_label=-1, reverse_decode=False):
        # feature_size = (src_embed_size * (3 + detail_dim) + 6) * sequence_length
        feature_size = (src_embed_size * 3 + detail_dim // 4 + 6) * sequence_length
        super().__init__(
            type_embed=L.EmbedID(type_size, src_embed_size, ignore_label),
            player_embed=L.EmbedID(player_size, src_embed_size, ignore_label),
            team_embed=L.EmbedID(team_size, src_embed_size, ignore_label),
            # detail_embed=L.EmbedID(detail_size, src_embed_size, ignore_label),
            detail_embed=L.Linear(detail_dim, detail_dim // 4),
            detail_bn=L.BatchNormalization(detail_dim),
            encoder=MLPEncoder(feature_size, hidden_size, mlp_layers),
            decoder=LSTMDecoder(vocab_size, trg_embed_size, hidden_size,
                                dropout_ratio, ignore_label)
        )
        self.embed_size = src_embed_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.start_token_id = end_token_id if reverse_decode else start_token_id
        self.end_token_id = start_token_id if reverse_decode else end_token_id
        self.reverse_decode = reverse_decode
        self.max_length = max_length
        self.detail_dim = detail_dim

    @staticmethod
    def preprocess(source_inputs):
        batch_size = source_inputs[0].shape[0]
        sequence_length = len(source_inputs)
        return batch_size, sequence_length

    def loss(self, source_inputs, target_inputs):
        # preprocessing
        batch_size, _ = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        loss, y_batch = self.decode_train(target_inputs, h, batch_size)
        return loss, y_batch

    def inference(self, source_inputs):
        # preprocessing
        batch_size, _ = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        y_hypo = self.decode_inference(h, batch_size, self.max_length)
        return y_hypo

    def embed(self, source_inputs, batch_size):
        # detail_shape = (batch_size, self.detail_dim * self.embed_size)
        embeded_inputs = []
        for x in source_inputs:
            embeded_type = self.type_embed(x[:, 0])
            embeded_player = self.player_embed(x[:, 1])
            embeded_team = self.team_embed(x[:, 2])
            embeded_detail = self.detail_embed(
                self.detail_bn(x[:, 9:].data.astype(self.xp.float32)))
            # embeded_detail = F.reshape(self.detail_embed(x[:, 9:]), detail_shape)
            embeded_inputs.append(F.concat((embeded_type, embeded_player, embeded_team,
                                            embeded_detail, x[:, 3:9].data.astype(self.xp.float32)), axis=1))
        return F.concat(embeded_inputs, axis=1)

    def encode(self, embeded_inputs):
        return self.encoder(embeded_inputs)

    def decode_train(self, target_inputs, h, batch_size):
        y, m = self.decoder.initialize(batch_size, self.start_token_id)
        s = h
        y_batch = []
        loss = 0
        for t in target_inputs:
            y, m, s = self.decoder(y, m, s, batch_size)
            y_batch.append(y.data.argmax(1).tolist())
            loss += F.softmax_cross_entropy(y, t)
            y = t
        return loss, y_batch

    def decode_inference(self, h, batch_size, max_length=20):
        y, m = self.decoder.initialize(batch_size, self.start_token_id)
        s = h
        y_hypo = []
        end_token_id = self.end_token_id
        xp = self.xp
        for _ in range(max_length):
            y, m, s = self.decoder(y, m, s, batch_size)
            p = y.data.argmax(1)
            if all(p == end_token_id):
                break
            y_hypo.append(p.tolist())
            y = chainer.Variable(p.astype(xp.int32))
        y_hypo = transpose(y_hypo)
        y_hypo = remove_eos(y_hypo, end_token_id)
        if self.reverse_decode:
            y_hypo = [y[::-1] for y in y_hypo]
        return y_hypo


class MLPEncoder2AttentionDecoder(MLPEncoder2LSTMDecoder):
    def __init__(self, type_size, player_size, team_size, detail_size, detail_dim,
                 src_embed_size, sequence_length, vocab_size, trg_embed_size,
                 hidden_size, start_token_id, end_token_id, softmax_class_weight,
                 mlp_layers=2, max_length=20, dropout_ratio=0.2, ignore_label=-1,
                 id_to_player=None, home_player_id=None, away_player_id=None,
                 id_to_team=None, home_team_id=None, away_team_id=None,
                 player_to_id=None, players=None, type_embed_size=None, reverse_decode=True):
        if type_embed_size is None:
            type_embed_size = src_embed_size
        # feature_size = (src_embed_size * (3 + detail_dim) + 6) * sequence_length
        scale = 4
        feature_size = (
            type_embed_size + src_embed_size * 2 + detail_dim // scale + 6) * sequence_length
        super(MLPEncoder2LSTMDecoder, self).__init__(
            type_embed=L.EmbedID(type_size, type_embed_size, ignore_label),
            player_embed=L.EmbedID(player_size, src_embed_size, ignore_label),
            team_embed=L.EmbedID(team_size, src_embed_size, ignore_label),
            # detail_embed=L.EmbedID(detail_size, src_embed_size, ignore_label),
            detail_embed=L.Linear(detail_dim, detail_dim // scale),
            detail_bn=L.BatchNormalization(detail_dim),
            encoder=MLPEncoder(feature_size, hidden_size, mlp_layers),
            # weight=L.Linear(hidden_size, hidden_size),
            decoder=AttentionDecoder(vocab_size, trg_embed_size,
                                     feature_size // sequence_length,
                                     hidden_size, dropout_ratio, ignore_label),
            # W_p=L.Linear(feature_size // sequence_length, len(player_to_id)),
        )
        self.embed_size = src_embed_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.start_token_id = end_token_id if reverse_decode else start_token_id
        self.end_token_id = start_token_id if reverse_decode else end_token_id
        self.reverse_decode = reverse_decode
        self.max_length = max_length
        self.detail_dim = detail_dim
        self.dropout_ratio = dropout_ratio
        self.softmax_class_weight = softmax_class_weight
        self.id_to_player = id_to_player
        self.player_to_id = player_to_id
        self.players = players
        self.home_player_id = home_player_id
        self.away_player_id = away_player_id
        self.player_id = [home_player_id, away_player_id]
        self.id_to_team = id_to_team
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.team_id = [home_team_id, away_team_id]
        self.mask_mode = bool(id_to_player) or bool(id_to_team)
        self.tag_loss = self.mask_mode

    def loss(self, source_inputs, target_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        loss, y_batch = self.decode_train(target_inputs, h, embeded_inputs,
                                          batch_size, sequence_length)
        return loss, y_batch

    def inference(self, source_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        y_hypo = self.decode_inference(h, batch_size, embeded_inputs, sequence_length,
                                       self.max_length, source_inputs)
        return y_hypo

    def convert_embed_inputs(self, embeded_inputs, h, batch_size, sequence_length):
        embeded_inputs = F.reshape(embeded_inputs,
                                   (sequence_length * batch_size,
                                    self.feature_size // sequence_length))
        return embeded_inputs

    def decode_train(self, target_inputs, h, embeded_inputs, batch_size, sequence_length,
                     player=None, team=None, source_inputs=None):
        y, m, s, c = self.decoder.initialize(batch_size, self.start_token_id, h, embeded_inputs,
                                             sequence_length)
        # s = self.weight(h)
        y_batch = []
        loss = 0
        class_weight = self.softmax_class_weight
        if source_inputs is not None:
            player_inputs = F.concat([x[:, 1][:, None] for x in source_inputs], axis=1)
            team_inputs = F.concat([x[:, 2][:, None] for x in source_inputs], axis=1)
        # for t in target_inputs:
        if not player and not team:
            player = team = [None] * len(target_inputs)
        # hyp_vec = ref_vec = 0
        for i, (t, p, tm) in enumerate(zip(
                target_inputs, player, team)):
            y, m, s, c = self.decoder(y, m, s, embeded_inputs, c, batch_size,
                                      sequence_length)
            y_batch.append(y.data.argmax(1).tolist())
            if p is not None and tm is not None and self.tag_loss:
                loss += self._compute_tag_loss(t.data, player_inputs, team_inputs, p, tm)
            loss += self._compute_word_loss(y, t, class_weight, i)
            y = t
        self.decoder_hidden = s
        return loss, y_batch

    def _compute_word_loss(self, y, t, class_weight, i):
        # loss = F.softmax_cross_entropy(y, t, class_weight=class_weight, reduce='no')
        # player_mask = self.xp.bitwise_or(
        #     t.data == self.home_team_id, t.data == self.away_player_id)
        # player_fill_value = 1
        # other_fill_value = 5 if i == 0 else 5
        # if player_mask.any():
        #     t_shape = t.shape
        #     player_weight = chainer.Variable(
        #         self.xp.full(t_shape, player_fill_value, dtype=self.xp.float32))
        #     other_weight = chainer.Variable(
        #         self.xp.full(t_shape, other_fill_value, dtype=self.xp.float32))
        #     scaler = F.where(player_mask, player_weight, other_weight)
        #     loss *= scaler
        # return F.sum(loss)

        # keyword_mask = reduce(self.xp.bitwise_or,
        #                       [t.data == i for i in self.keyword_ids])
        # if keyword_mask.any():
        #     t_shape = t.shape
        #     keyword_weight = chainer.Variable(
        #         self.xp.full(t_shape, 5, dtype=self.xp.float32))
        #     other_weight = chainer.Variable(
        #         self.xp.full(t_shape, 1, dtype=self.xp.float32))
        #     scaler = F.where(keyword_mask, keyword_weight, other_weight)
        #     loss *= scaler
        # return F.sum(loss)
        return F.softmax_cross_entropy(y, t, class_weight=class_weight, reduce='mean')

    def _compute_tag_loss(self, indices, player_inputs, team_inputs,
                          player_targets, team_targets):
        xp = self.xp
        loss = 0
        player_tag = reduce(xp.bitwise_or, [indices == i for i in self.player_id])
        player_tag = xp.bitwise_and(player_tag, indices != -1)
        team_tag = reduce(xp.bitwise_or, [indices == i for i in self.team_id])
        team_tag = xp.bitwise_and(team_tag, indices != -1)
        zeros = chainer.Variable(xp.zeros(self.decoder.a.shape, dtype=xp.float32))
        if xp.any(player_tag):
            mask = F.broadcast_to(player_tag[:, None], player_inputs.shape)
            player_targets = F.broadcast_to(player_targets.data[:, None],
                                            player_inputs.shape)
            # log
            # log_score = F.log(self.decoder.a)
            # log_loss = F.where(player_inputs.data == player_targets.data,
            #                    log_score, - log_score)
            # loss += F.sum(F.where(xp.isinf(log_loss.data), zeros, log_loss))
            # no log
            tmp_loss = F.where(player_inputs.data == player_targets.data,
                               zeros, self.decoder.a)
            loss += F.sum(F.where(mask, tmp_loss, zeros))
        if xp.any(team_tag):
            mask = F.broadcast_to(team_tag[:, None], player_inputs.shape)
            team_targets = F.broadcast_to(team_targets.data[:, None],
                                          team_inputs.shape)
            # log
            # log_score = F.log(self.decoder.a)
            # log_loss = F.where(team_inputs.data == team_targets.data,
            #                    log_score, - log_score)
            # log_loss = F.where(mask, log_loss, zeros)
            # loss += F.sum(F.where(xp.isinf(log_loss.data), zeros, log_loss))
            tmp_loss = F.where(team_inputs.data == team_targets.data,
                               zeros, self.decoder.a)
            loss += F.sum(F.where(mask, tmp_loss, zeros))
        return loss

    def decode_inference(self, h, batch_size, embeded_inputs, sequence_length, max_length=20,
                         source_inputs=None):
        y, m, s, c = self.decoder.initialize(batch_size, self.start_token_id, h, embeded_inputs,
                                             sequence_length)
        # s = self.weight(h)
        y_hypo = []
        end_token_id = self.end_token_id
        xp = self.xp
        # player_inputs.shape = (event_length, batch_size)
        player_inputs = [x.data[:, 1].tolist() for x in source_inputs]
        team_inputs = [x.data[:, 2].tolist() for x in source_inputs]
        for _ in range(max_length):
            y, m, s, c = self.decoder(y, m, s, embeded_inputs, c, batch_size,
                                      sequence_length)
            p = y.data.argmax(1)
            p_list = p.tolist()
            if self.mask_mode:
                p_cpu = cuda.to_cpu(p)
                # a.shape = (batch_size,)
                a = self.decoder.a.data.argmax(1).tolist()
                player_ids = self._get_id(a, player_inputs, 'player{}', self.id_to_player)
                team_ids = self._get_id(a, team_inputs, 'team{}', self.id_to_team)
                player_mask = reduce(np.bitwise_or, [p_cpu == i for i in self.player_id])
                p_list = np.where(player_mask, player_ids, p_list)
                team_mask = reduce(np.bitwise_or, [p_cpu == i for i in self.team_id])
                p_list = np.where(team_mask, team_ids, p_list)
                # player = self.W_p(c).data.argmax(1).tolist()
                # p_list = ['player{}'.format(
                #           self.players[player[i] if player[i] < len(self.players) else 0])
                #           if y == self.home_player_id or y == self.away_player_id else y
                #           for i, y in enumerate(p_list)]
            if all(p == end_token_id):
                break
            y_hypo.append([str(y) for y in p_list])
            y = chainer.Variable(p.astype(xp.int32))
        y_hypo = transpose(y_hypo)
        y_hypo = remove_eos(y_hypo, str(end_token_id))
        if self.reverse_decode:
            y_hypo = [y[::-1] for y in y_hypo]
        return y_hypo

    def _get_id(self, indices, id_list, prefix, id_to_token):
        ids = [prefix.format(id_to_token[id_list[ind][batch]])
               for batch, ind in enumerate(indices)]
        return ids


class MLPEncoder2GatedAttentionDecoder(MLPEncoder2AttentionDecoder):
    def __init__(self, type_size, player_size, team_size, detail_size,
                 detail_dim, src_embed_size, sequence_length, vocab_size,
                 trg_embed_size, hidden_size, start_token_id, end_token_id,
                 softmax_class_weight, mlp_layers=2, max_length=20,
                 dropout_ratio=0.2, ignore_label=-1, id_to_player=None,
                 home_player_id=None, away_player_id=None, id_to_team=None,
                 home_team_id=None, away_team_id=None, player_to_id=None, players=None,
                 type_embed_size=None, reverse_decode=False):
        super().__init__(type_size, player_size, team_size, detail_size,
                         detail_dim, src_embed_size, sequence_length, vocab_size,
                         trg_embed_size, hidden_size, start_token_id, end_token_id,
                         softmax_class_weight, mlp_layers, max_length,
                         dropout_ratio, ignore_label, id_to_player, home_player_id,
                         away_player_id, id_to_team, home_team_id, away_team_id,
                         player_to_id, players, type_embed_size, reverse_decode)
        # self.add_link('W_g', L.Linear(hidden_size + self.feature_size // sequence_length, 1))
        self.add_link('reconstructor',
                      MLPEncoder(hidden_size, self.feature_size, mlp_layers))

    def convert_embed_inputs(self, embeded_inputs, h, batch_size, sequence_length):
        # embeded_inputs.shape = (batch_size, feature_size)
        # feature_size = sequence_length * embed_size
        embeded_inputs = super().convert_embed_inputs(embeded_inputs, h, batch_size,
                                                      sequence_length)
        # embeded_inputs.shape = (batch_size * sequence_length, embed_size)
        batch_x_sequence, embed_size = embeded_inputs.shape
        # h_expand = F.reshape(F.broadcast_to(
        #     F.expand_dims(h, axis=1), (batch_size, sequence_length, self.hidden_size)),
        #     (batch_x_sequence, self.hidden_size))
        # gate_in = F.concat((h_expand, embeded_inputs), axis=1)
        # gate = F.broadcast_to(F.sigmoid(self.W_g(gate_in)), (batch_x_sequence, embed_size))
        h = F.reshape(F.sigmoid(self.reconstructor(h)), (batch_x_sequence, embed_size))
        return embeded_inputs * h


# class MLPEncoder2EachGatedAttentionDecoder(MLPEncoder2AttentionDecoder):
#     def __init__(self, type_size, player_size, team_size, detail_size,
#                  detail_dim, src_embed_size, sequence_length, vocab_size,
#                  trg_embed_size, hidden_size, start_token_id, end_token_id,
#                  softmax_class_weight, gate_size, mlp_layers=2, max_length=20,
#                  dropout_ratio=0.2, ignore_label=-1, id_to_player=None,
#                  home_player_id=None, away_player_id=None, id_to_team=None,
#                  home_team_id=None, away_team_id=None):
#         super().__init__(type_size, player_size, team_size, detail_size,
#                          detail_dim, src_embed_size, sequence_length, vocab_size,
#                          trg_embed_size, hidden_size, start_token_id, end_token_id,
#                          softmax_class_weight, mlp_layers, max_length,
#                          dropout_ratio, ignore_label, id_to_player, home_player_id,
#                          away_player_id, id_to_team, home_team_id, away_team_id)
#         self.add_link('W_g', L.Linear(hidden_size + self.feature_size // sequence_length,
#                                       len(gate_size)))
#         self.gate_size = gate_size
#
#     def convert_embed_inputs(self, embeded_inputs, h, batch_size, sequence_length):
#         # embeded_inputs.shape = (batch_size, feature_size)
#         # feature_size = sequence_length * embed_size
#         embeded_inputs = super().convert_embed_inputs(embeded_inputs, h, batch_size,
#                                                       sequence_length)
#         # embeded_inputs.shape = (batch_size * sequence_length, embed_size)
#         batch_x_sequence, _ = embeded_inputs.shape
#         h = F.reshape(F.broadcast_to(F.expand_dims(h, axis=1),
#                                      (batch_size, sequence_length, self.hidden_size)),
#                       (batch_x_sequence, self.hidden_size))
#         gate_in = F.concat((h, embeded_inputs), axis=1)
#         # (batch x seq, gate_size)
#         tmp_gate = F.sigmoid(self.W_g(gate_in))
#         gate = F.concat([F.broadcast_to(tmp_gate[:, i][:, None], (batch_x_sequence, dim))
#                          for i, dim in self.gate_size], axis=1)
#         return embeded_inputs * gate


class DiscriminativeMLPEncoder2AttentionDecoder(MLPEncoder2AttentionDecoder):
    def __init__(self, type_size, player_size, team_size, detail_size,
                 detail_dim, src_embed_size, sequence_length, vocab_size, content_word_vocab_size,
                 trg_embed_size, hidden_size, start_token_id, end_token_id,
                 softmax_class_weight, sigmoid_class_weight, loss_weight, loss_type='ce',
                 mlp_layers=2, max_length=20, dropout_ratio=0.2, ignore_label=-1,
                 id_to_player=None, home_player_id=None, away_player_id=None,
                 id_to_team=None, home_team_id=None, away_team_id=None,
                 player_to_id=None, players=None, type_embed_size=None,
                 reverse_decode=False):
        super().__init__(type_size, player_size, team_size, detail_size,
                         detail_dim, src_embed_size, sequence_length, vocab_size,
                         trg_embed_size, hidden_size, start_token_id, end_token_id,
                         softmax_class_weight, mlp_layers, max_length, dropout_ratio,
                         ignore_label, id_to_player, home_player_id, away_player_id,
                         id_to_team, home_team_id, away_team_id, player_to_id, players,
                         type_embed_size, reverse_decode)
        self.sigmoid_class_weight = sigmoid_class_weight
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.add_link('discriminative', L.Linear(hidden_size, content_word_vocab_size))

    def loss(self, source_inputs, target_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        text, label, player, team = target_inputs
        loss, y_batch = self.decode_train(text, h, embeded_inputs, batch_size,
                                          sequence_length, player, team, source_inputs)
        y = self.discriminative(h)
        if self.loss_type == 'mse':
            loss += F.mean_squared_error(F.sigmoid(y),
                                         chainer.Variable(label.data.astype(self.xp.float32))
                                         ) * self.loss_weight
        elif self.loss_type == 'ce':
            loss += weighted_sigmoid_cross_entropy(
                y, label, class_weight=self.sigmoid_class_weight
            ) * self.loss_weight
        return loss, y_batch

    def inference(self, source_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        y_hypo = self.decode_inference(h, batch_size, embeded_inputs, sequence_length,
                                       self.max_length, source_inputs)
        return y_hypo, F.sigmoid(self.discriminative(h))


class DiscriminativeMLPEncoder2GatedAttentionDecoder(MLPEncoder2GatedAttentionDecoder):
    def __init__(self, type_size, player_size, team_size, detail_size,
                 detail_dim, src_embed_size, sequence_length, vocab_size, content_word_vocab_size,
                 trg_embed_size, hidden_size, start_token_id, end_token_id,
                 softmax_class_weight, sigmoid_class_weight, loss_weight, loss_type='ce',
                 mlp_layers=2, max_length=20, dropout_ratio=0.2, ignore_label=-1,
                 id_to_player=None, home_player_id=None, away_player_id=None,
                 id_to_team=None, home_team_id=None, away_team_id=None,
                 player_to_id=None, players=None, type_embed_size=None,
                 reverse_decode=False):
        super().__init__(type_size, player_size, team_size, detail_size,
                         detail_dim, src_embed_size, sequence_length, vocab_size,
                         trg_embed_size, hidden_size, start_token_id, end_token_id,
                         softmax_class_weight, mlp_layers, max_length, dropout_ratio,
                         ignore_label, id_to_player, home_player_id, away_player_id,
                         id_to_team, home_team_id, away_team_id, player_to_id, players,
                         type_embed_size, reverse_decode)
        self.sigmoid_class_weight = sigmoid_class_weight
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.add_link('discriminative', L.Linear(hidden_size, content_word_vocab_size))
        # self.add_link('discriminative2', L.Linear(hidden_size, content_word_vocab_size))

    def loss(self, source_inputs, target_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        text, label, player, team = target_inputs
        loss, y_batch = self.decode_train(text, h, embeded_inputs, batch_size,
                                          sequence_length, player, team, source_inputs)
        y = self.discriminative(h)
        if self.loss_type == 'mse':
            loss += F.mean_squared_error(F.sigmoid(y),
                                         chainer.Variable(label.data.astype(self.xp.float32))
                                         ) * self.loss_weight
        elif self.loss_type == 'ce':
            loss += weighted_sigmoid_cross_entropy(
                y, label, class_weight=self.sigmoid_class_weight
            ) * self.loss_weight
        # y = self.discriminative2(self.decoder_hidden)
        # loss += weighted_sigmoid_cross_entropy(
        #             y, label, class_weight=self.sigmoid_class_weight
        #         ) * self.loss_weight
        return loss, y_batch

    def inference(self, source_inputs):
        # preprocessing
        batch_size, sequence_length = self.preprocess(source_inputs)
        # embeding
        embeded_inputs = self.embed(source_inputs, batch_size)
        # encoding
        h = self.encode(embeded_inputs)
        # decoding
        embeded_inputs = self.convert_embed_inputs(embeded_inputs, h,
                                                   batch_size, sequence_length)
        y_hypo = self.decode_inference(h, batch_size, embeded_inputs, sequence_length,
                                       self.max_length, source_inputs)
        return y_hypo, F.sigmoid(self.discriminative(h))


class DiscriminativeGLUEncoder2GatedAttentionDecoder(
        DiscriminativeMLPEncoder2GatedAttentionDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.encoder
        self.add_link('encoder', GLUEncoder(self.embed_size, self.hidden_size))

    def embed(self, source_inputs, batch_size):
        embeded_inputs = super().embed(source_inputs, batch_size)
        batch_size, hidden_size = embeded_inputs.shape
        sequence_length = len(source_inputs)
        feature_size = hidden_size // sequence_length
        return F.reshape(embeded_inputs, (batch_size, 1, feature_size, sequence_length))

    def convert_embed_inputs(self, embeded_inputs, h, batch_size, sequence_length):
        batch_size, _, feature_size, sequence_length = embeded_inputs.shape
        return super().convert_embed_inputs(
            F.reshape(embeded_inputs, (batch_size * sequence_length, feature_size)),
            h, batch_size, sequence_length
        )


class DiscriminativeMLPEncoder2GatedAttentionDecoderWithGateLoss(
        DiscriminativeMLPEncoder2GatedAttentionDecoder):
    def loss(self, source_inputs, target_inputs):
        loss, y_batch = super().loss(source_inputs, target_inputs)
        gate = self.gate
        batch_size = source_inputs[0].shape[0]
        batch_x_sequence, _ = gate.shape
        sequence_length = batch_x_sequence // batch_size
        gate = F.reshape(gate, (batch_size, sequence_length))
        zeros = chainer.Variable(self.xp.zeros((batch_size, sequence_length),
                                               dtype=self.xp.float32))
        gate = F.where(gate.data > 0.5, gate, zeros)
        loss += F.sum((F.sum(gate, axis=1) - 1) ** 2)
        return loss, y_batch

# class DiscriminativeMLPEncoder2EachGatedAttentionDecoder(
#         DiscriminativeMLPEncoder2GatedAttentionDecoder):
#     def __init__(self, type_size, player_size, team_size, detail_size,
#                  detail_dim, src_embed_size, sequence_length, vocab_size, content_word_vocab_size,
#                  trg_embed_size, hidden_size, start_token_id, end_token_id,
#                  softmax_class_weight, sigmoid_class_weight, loss_weight, loss_type,
#                  gate_size, mlp_layers=2, max_length=20,
#                  dropout_ratio=0.2, ignore_label=-1):
#         super().__init__(type_size, player_size, team_size, detail_size,
#                          detail_dim, src_embed_size, sequence_length, vocab_size,
#                          content_word_vocab_size, trg_embed_size, hidden_size,
#                          start_token_id, end_token_id, softmax_class_weight,
#                          sigmoid_class_weight, loss_weight, loss_type, mlp_layers,
#                          max_length, dropout_ratio, ignore_label)
#         del self.W_g
#         self.add_link('W_g', L.Linear(hidden_size + self.feature_size // sequence_length,
#                                       len(gate_size)))
#         self.gate_size = gate_size
#
#     def convert_embed_inputs(self, embeded_inputs, h, batch_size, sequence_length):
#         embeded_inputs = super().convert_embed_inputs(embeded_inputs, h, batch_size,
#                                                       sequence_length)
#         batch_x_sequence, _ = embeded_inputs.shape
#         h = F.reshape(F.broadcast_to(F.expand_dims(h, axis=1),
#                                      (batch_size, sequence_length, self.hidden_size)),
#                       (batch_x_sequence, self.hidden_size))
#         gate_in = F.concat((h, embeded_inputs), axis=1)
#         tmp_gate = F.sigmoid(self.W_g(gate_in))
#         gate = F.concat([F.broadcast_to(tmp_gate[:, i][:, None], (batch_x_sequence, dim))
#                          for i, dim in self.gate_size], axis=1)
#         return embeded_inputs * gate
#
#
# class DiscriminativeMLPEncoder2GatedEachAttentionDecoder(
#         DiscriminativeMLPEncoder2GatedAttentionDecoder):
#     def __init__(self, type_size, player_size, team_size, detail_size,
#                  detail_dim, src_embed_size, sequence_length, vocab_size, content_word_vocab_size,
#                  trg_embed_size, hidden_size, start_token_id, end_token_id,
#                  softmax_class_weight, sigmoid_class_weight, loss_weight, loss_type,
#                  attention_size, mlp_layers=2, max_length=20,
#                  dropout_ratio=0.2, ignore_label=-1):
#         super().__init__(type_size, player_size, team_size, detail_size,
#                          detail_dim, src_embed_size, sequence_length, vocab_size,
#                          content_word_vocab_size, trg_embed_size, hidden_size,
#                          start_token_id, end_token_id, softmax_class_weight,
#                          sigmoid_class_weight, loss_weight, loss_type, mlp_layers, max_length,
#                          dropout_ratio, ignore_label)
#         del self.decoder
#         self.add_link('decoder', EachAttentionDecoder(vocab_size, trg_embed_size,
#                                                       self.feature_size // sequence_length,
#                                                       hidden_size, attention_size,
#                                                       dropout_ratio, ignore_label))
