import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from mltools.model import BaseModel


class LSTMEncoder(BaseModel):
    def __init__(self, embed_size, hidden_size, dropout_ratio=0.2):
        super().__init__(
            # input weight vector of {input, output, forget} gate and input
            W_e=L.Linear(embed_size, 4 * hidden_size),
            # hidden weight vector of {input, output, forget} gate and input
            W_h=L.Linear(hidden_size, 4 * hidden_size),
        )
        self.dropout_ratio = dropout_ratio

    def __call__(self, embeded_x, m_prev, h_prev, feed_previous):
        lstm_in = F.dropout(self.W_e(embeded_x) + self.W_h(h_prev),
                            ratio=self.dropout_ratio)
        m_tmp, h_tmp = F.lstm(m_prev, lstm_in)
        m = F.where(feed_previous, m_prev, m_tmp)
        h = F.where(feed_previous, h_prev, h_tmp)
        return m, h


class MLPBlock(BaseModel):
    def __init__(self, input_size, output_size):
        super().__init__(
            bn=L.BatchNormalization(input_size),
            fc=L.Linear(input_size, output_size),
        )

    def __call__(self, x):
        return F.relu(self.fc(self.bn(x)))


class MLPEncoder(BaseModel):
    def __init__(self, input_size, output_size, layers=2):
        hidden_size = int((input_size + output_size) * 2 / 3)
        super().__init__()
        self.add_link('input', MLPBlock(input_size, hidden_size))
        self.blocks = ['fc{}'.format(i) for i in range(layers)]
        [self.add_link(block, MLPBlock(hidden_size, hidden_size))
            for block in self.blocks]
        self.add_link('output', L.Linear(hidden_size, output_size))

    def __call__(self, embeded_x):
        h = self.input(embeded_x)
        for block in self.blocks:
            h = self[block](h) + h
        return self.output(h)


def add_zero_pad(x, width, axis):
    xp = cuda.get_array_module(x)
    shape = list(x.shape)
    shape[axis] = width
    zero_mat = chainer.Variable(xp.zeros(shape, dtype=x.dtype))
    return F.concat((zero_mat, x), axis=axis)


class GLU(BaseModel):
    def __init__(self, in_channel, out_channel, kernel):
        super().__init__(
            W=L.Convolution2D(in_channel, out_channel, (1, kernel), 1, 0),
            V=L.Convolution2D(in_channel, out_channel, (1, kernel), 1, 0)
        )
        self.kernel = kernel

    def __call__(self, x):
        x = add_zero_pad(x, self.kernel // 2, 3)
        A = F.tanh(self.W(x))
        B = F.sigmoid(self.V(x))
        return A * B


class ResBlock(BaseModel):
    def __init__(self, in_channel, out_channel, kernel, block):
        super().__init__()
        links = []
        for i in range(block):
            links.append(('glu{}'.format(i), GLU(in_channel, out_channel, kernel)))
            in_channel = out_channel
        [self.add_link(*l) for l in links]
        self.links = links

    def __call__(self, x):
        h = x
        for name, _ in self.links:
            h = getattr(self, name)(h)
        channel_diff = h.shape[1] - x.shape[1]
        length_diff = x.shape[3] - h.shape[3]
        return add_zero_pad(h, length_diff, 3) + add_zero_pad(x, channel_diff, 1)


class GLUEncoder(BaseModel):
    def __init__(self, embed_size, hidden_size):
        super().__init__(
            block1=ResBlock(1, 16, 4, 2),
            block2=ResBlock(16, 16, 4, 2),
            conv=L.Convolution2D(16, hidden_size, (1, 4), 1, 0)
        )
        self.hidden_size = hidden_size

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.conv(add_zero_pad(h, 4 // 2, 3))
        batch, channel, height, width = h.shape
        return F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channel))
