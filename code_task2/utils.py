import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class EMA:
    """
        REF: https://zhuanlan.zhihu.com/p/68748778
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def util_loss_fct(logits, labels, sample_weights, num_labels, loss_name=None):
    # logits = [batch,class_num], labels = [batch], sample_weights = [batch]
    if loss_name == 'ce':
        loss_fct = CrossEntropyLoss(reduction='none')    # 2022.05.04 添加sample-weight
        ce_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        ce_loss = torch.sum(ce_loss * sample_weights) / torch.sum(sample_weights)
        return ce_loss

    if loss_name == 'ce-soft-label':
        probs = F.log_softmax(logits, dim=1)
        ce_loss = -torch.sum(labels * probs, dim=1)
        ce_loss = torch.sum(ce_loss * sample_weights) / torch.sum(sample_weights)
        return ce_loss

    if loss_name == 'focal':   # multi-class focal loss   F1=0.7319  落败于交叉熵
        gamma = 2.0
        loss_fct = CrossEntropyLoss(reduction='none')
        ce_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        # pt = torch.exp(-ce_loss)    # 方式1
        # pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1) # 方式2, labels is one-hot
        pt = torch.softmax(logits, dim=1).index_select(-1, labels)   # 方式3
        focal_loss = (1-pt)**gamma * ce_loss
        focal_loss = torch.sum(focal_loss * sample_weights) / torch.sum(sample_weights)
        return focal_loss

    if loss_name == 'ghm-c':      # multi-class focal loss   F1=0.7202  落败于交叉熵
        #  reference: https://github.com/shuxinyin/NLP-Loss-Pytorch/blob/master/unbalanced_loss/GHM_loss.py
        x = logits
        target = F.one_hot(labels, num_classes=num_labels)
        label_weight = torch.ones_like(target)
        ghm_loss_fct = GHMC()
        ghm_loss = ghm_loss_fct(logits, target, label_weight, reduction='none')
        ghm_loss = torch.sum(ghm_loss * sample_weights) / torch.sum(sample_weights)
        return ghm_loss

    if loss_name == 'ce-poly-1':
        # ICLR 2022, poly_loss
        epsilon = 1.0
        loss_fct = CrossEntropyLoss(reduction='none')
        ce_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        # pt = torch.exp(-ce_loss)    # 方式1
        # pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1) # 方式2, labels is one-hot
        # pt = torch.softmax(logits, dim=1).index_select(-1, labels)   # 错误实现, 方式3, index_select函数应该用gather替换
        pt = torch.sum(torch.softmax(logits, dim=1) * F.one_hot(labels, num_classes=num_labels), dim=1)
        poly1_loss = epsilon*(1-pt) + ce_loss
        poly1_loss = torch.sum(poly1_loss * sample_weights) / torch.sum(sample_weights)
        return poly1_loss

    if loss_name == 'r-drop-kl':
        # https://github.com/dropreg/R-Drop
        p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(labels, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(labels, dim=-1), F.softmax(logits, dim=-1), reduction='none')
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        kl_loss = p_loss + q_loss
        kl_loss = torch.sum(kl_loss * sample_weights) / torch.sum(sample_weights)
        return kl_loss


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"

    https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/losses/ghm_loss.html
    """

    def __init__(self,
                 bins=10,
                 momentum=0,
                 use_sigmoid=True,
                 loss_weight=1.0,
                 reduction='mean'):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                pred,
                target,
                label_weight,
                reduction_override=None,
                **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(
                target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        # loss = weight_reduce_loss(                             # 2022.05.10 注释原始代码。reduction=None, do nothing
        #     loss, weights, reduction=reduction, avg_factor=tot)
        loss = loss * weights

        loss = torch.sum(loss, dim=1)       # 2022.05.10 新加代码。multi-class ghm loss能直接在这求和？

        return loss * self.loss_weight

def get_vocab_from_file(file_path):
    idx = 0
    name_idx = dict()
    with open(file_path, mode='r') as fin:
        for line in fin:
            item = line.strip()
            if item != '[UNK]':
                name_idx[item] = idx
            idx += 1
    # print( len(name_idx.keys()) )
    return name_idx

class FGM():
    """
        知乎-瓦特兰蒂斯
        https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}