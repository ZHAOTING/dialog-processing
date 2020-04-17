import code

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelSmoothingCrossEntropyLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, num_labels=0, ignore_index=-1, reduction='batchmean'):
        assert 0.0 < label_smoothing <= 1.0
        self.confidence = 1.0 - label_smoothing
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        super(LabelSmoothingCrossEntropyLoss, self).__init__(reduction=reduction)

        assert label_smoothing > 0
        assert num_labels > 0

        smoothing_value = label_smoothing / (num_labels - 2)
        one_hot = torch.full((num_labels,), smoothing_value).to(DEVICE)
        if self.ignore_index >= 0:
            one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        """forward

        Arguments:
            output {FloatTensor [n_inputs * n_classes]}
            target {LongTensor [n_inputs]} -- NOTE: target has to be non-negative

        Returns:
            {FloatTensor} -- see torch.CrossEntropyLoss
        """
        output = F.log_softmax(output, -1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)

        # filter ignored pairs
        mask = (target != self.ignore_index).unsqueeze(1)
        filtered_output = output.masked_select(mask).view(-1, self.num_labels)
        filtered_model_prob = model_prob.masked_select(mask).view(-1, self.num_labels)

        return F.kl_div(filtered_output, filtered_model_prob, reduction=self.reduction)
