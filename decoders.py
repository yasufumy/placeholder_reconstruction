import chainer
import chainer.functions as F
import chainer.links as L
from mltools.model import BaseModel


class LSTMDecoder(BaseModel):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 dropout_ratio=0.2, ignore_label=-1):
        super().__init__(
            # Weights of Decoder
            E=L.EmbedID(vocab_size, embed_size, ignore_label),
            # input_size will be changed if you use input feeding
            W_e=L.Linear(None, 4 * hidden_size),
            W_s=L.Linear(hidden_size, 4 * hidden_size),
            # input_size will be changed if you use attention
            W_o=L.Linear(None, vocab_size),
        )
        self.decoder_hidden_size = hidden_size
        self.ignore_label = ignore_label
        self.dropout_ratio = dropout_ratio

    def __call__(self, y, m_prev, s_prev, batch_size):
        embeded_y = self.E(y)
        m, s = self.step(y, embeded_y, m_prev, s_prev, batch_size)
        return self.W_o(s), m, s

    def step(self, y, embeded_y, m_prev, s_prev, batch_size):
        # decode once
        lstm_in = F.dropout(self.W_e(embeded_y) + self.W_s(s_prev),
                            ratio=self.dropout_ratio)
        m_tmp, s_tmp = F.lstm(m_prev, lstm_in)
        feed_previous = F.broadcast_to(F.expand_dims(y.data == self.ignore_label, -1),
                                       (batch_size, self.decoder_hidden_size))
        m = F.where(feed_previous, m_prev, m_tmp)
        s = F.where(feed_previous, s_prev, s_tmp)
        return m, s

    def initialize(self, batch_size, start_token_id):
        xp = self.xp
        initial_inputs = xp.full((batch_size,), start_token_id, dtype=xp.int32)
        initial_state = xp.zeros((batch_size, self.decoder_hidden_size), dtype=xp.float32)
        return chainer.Variable(initial_inputs), chainer.Variable(initial_state)


class AttentionDecoder(LSTMDecoder):
    def __init__(self, vocab_size, embed_size,
                 encoder_hidden_size, decoder_hidden_size,
                 dropout_ratio=0.2, ignore_label=-1):
        super().__init__(vocab_size, embed_size, decoder_hidden_size,
                         dropout_ratio, ignore_label)
        self.add_link('W1', L.Linear(encoder_hidden_size, decoder_hidden_size))
        self.add_link('W2', L.Linear(decoder_hidden_size, decoder_hidden_size))
        self.add_link('v', L.Linear(decoder_hidden_size, 1))
        self.encoder_hidden_size = encoder_hidden_size

    def _attention(self, h, s, batch_size, sequence_length):
        decoder_hidden_size = self.decoder_hidden_size
        encoder_hidden_size = self.encoder_hidden_size

        input_shape = (batch_size, sequence_length, decoder_hidden_size)
        weighted_h = F.reshape(self.W1(h), input_shape)
        weighted_s = F.broadcast_to(F.expand_dims(self.W2(s), axis=1), input_shape)

        score = self.v(F.reshape(F.tanh(weighted_s + weighted_h),
                                 (batch_size * sequence_length, decoder_hidden_size)))
        a = F.softmax(F.reshape(score, (batch_size, sequence_length)))
        self.a = a
        # c = F.matmul(F.reshape(h, (batch_size, encoder_hidden_size, sequence_length)),
        #              a[..., None])
        c = F.batch_matmul(F.reshape(h, (batch_size, encoder_hidden_size, sequence_length)), a)
        return F.reshape(c, (batch_size, encoder_hidden_size))

    def __call__(self, y, m_prev, s_prev, h, c, batch_size, sequence_length):
        embeded_y = F.concat((self.E(y), c), axis=1)
        # m is memory cell of lstm, s is previous hidden output
        # decode once
        m, s = self.step(y, embeded_y, m_prev, s_prev, batch_size)
        # calculate attention
        c = self._attention(h, s, batch_size, sequence_length)
        return self.W_o(F.concat((s, c), axis=1)), m, s, c

    def initialize(self, batch_size, start_token_id, encoder_output, h,
                   sequence_length):
        initial_inputs, initial_state = super().initialize(batch_size, start_token_id)
        initial_context = chainer.Variable(self.xp.zeros((batch_size,
                                                          self.encoder_hidden_size),
                                                         dtype=self.xp.float32))
        return initial_inputs, initial_state, initial_state, initial_context


class EachAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size,
                 encoder_hidden_size, decoder_hidden_size, attention_size,
                 dropout_ratio=0.2, ignore_label=-1):
        super().__init__(vocab_size, embed_size, encoder_hidden_size, decoder_hidden_size,
                         dropout_ratio, ignore_label)
        del self.v
        self.add_link('v', L.Linear(decoder_hidden_size, len(attention_size)))
        self.attention_size = attention_size

    def _attention(self, h, s, batch_size, sequence_length):
        decoder_hidden_size = self.decoder_hidden_size
        encoder_hidden_size = self.encoder_hidden_size

        input_shape = (batch_size, sequence_length, decoder_hidden_size)
        weighted_h = F.reshape(self.W1(h), input_shape)
        weighted_s = F.broadcast_to(F.expand_dims(self.W2(s), axis=1), input_shape)

        score = self.v(F.reshape(F.tanh(weighted_s + weighted_h),
                                 (batch_size * sequence_length, decoder_hidden_size)))
        a = F.softmax(F.reshape(score, (batch_size, len(self.attention_size), sequence_length)),
                      axis=2)
        score = F.concat([F.broadcast_to(a[:, i, :][:, None, :],
                                         (batch_size, dim, sequence_length))
                          for i, dim in self.attention_size], axis=1)
        c = F.sum(F.reshape(h, (batch_size, encoder_hidden_size, sequence_length)) * score,
                  axis=2)
        return F.reshape(c, (batch_size, encoder_hidden_size))
