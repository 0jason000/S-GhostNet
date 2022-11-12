
"""define loss function for network."""

from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn


class LabelSmoothingCrossEntropy(_Loss):
    """cross-entropy with label smoothing"""

    def __init__(self, smooth_factor=0.1, num_classes=1000):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        self.cast = P.Cast()

    def construct(self, logits, label):
        label = self.cast(label, mstype.int32)
        one_hot_label = self.onehot(label, F.shape(
            logits)[1], self.on_value, self.off_value)
        loss_logit = self.ce(logits, one_hot_label)
        return loss_logit
      