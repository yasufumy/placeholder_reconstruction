from chainer import functions as F
from chainer import Variable
from chainer import Function


def weighted_sigmoid_cross_entropy(x, t, normalize=False, reduce='mean', class_weight=None):
    if class_weight is None:
        return F.sigmoid_cross_entropy(x, t, normalize, reduce)
    else:
        loss = F.sigmoid_cross_entropy(x, t, normalize, reduce='no')
        mask = Variable(t.data == 0)
        loss = F.where(mask, loss * class_weight[0], loss * class_weight[1])
        if reduce == 'mean':
            return F.sum(loss) / len(loss)
        else:
            return loss


def binary_each_accuracy(y, t):
    y = y.data.ravel()
    t = t.data.ravel()
    pos_count = (t == 1).sum()
    neg_count = (t == 0).sum()
    pred = (y >= 0.5)
    y_pos = (pred[t == 1] == 1).sum()
    y_neg = (pred[t == 0] == 0).sum()
    return (y_pos / pos_count).tolist(), (y_neg / neg_count).tolist()


class GradScaler(Function):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, x):
        return x

    def backward(self, x, gy):
        gw, = gy
        return gw / self.scale,


def gradscaler(x, scale):
    return GradScaler(scale)(x)
