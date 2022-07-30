import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.utils import ModelOutput
from transformers.activations import gelu
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaClassificationHead

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
            if param.requires_grad and 'encoder_k_model' not in name:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'encoder_k_model' not in name :
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'encoder_k_model' not in name:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'encoder_k_model' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class SimCseMoCoClassifier(nn.Module):
    """
    1. Moco
        https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    2. InfoXLM
        https://github.com/microsoft/unilm/blob/master/infoxlm/src-infoxlm/infoxlm/criterions/xlco.py
    """
    def __init__(self, encoder_q = None, config = None, dim=1024, T=0.05, K=int(8*2.5), M=0.999):      # base=768  large=1024
        super(SimCseMoCoClassifier, self).__init__()
        """
            dim: feature dimension (default: 768)
            K: queue size; number of negative keys (default: 2.5*batchsize, From ESimCSE)
            m: moco momentum of updating key encoder (default: 0.999)
            T: softmax temperature (default: 0.05)
        """
        self.queue_size = K
        self.M = M
        self.T = T
        self.scale = 20.0   # 1/T
        self.dim = dim
        self.encoder_q = encoder_q          # 浅拷贝,应该会一直指向原始模型数据
        self.encoder_k = RobertaModel(config, add_pooling_layer=False)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(self.queue_size, self.dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.M + param_q.data * (1. - self.M)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
            https://github.com/microsoft/unilm/blob/master/infoxlm/src-infoxlm/infoxlm/models/infoxlm.py#L78
        """
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr+batch_size, :] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            left_len = self.queue_size - ptr
            self.queue[ptr:, :] = keys[:left_len, :]
            ptr = batch_size-left_len
            self.queue[:ptr, :] = keys[left_len:, :]
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        sample_weights: Optional[torch.FloatTensor] = None,
    ):
        """
            无监督Simcse + Moco, 对比学习 Loss; 后续考虑使用debiased-contrasitive
        """
        with torch.no_grad():
            # 1. update K Encoder
            self._momentum_update_key_encoder()
            # 2. get Emb from K
            output_features = self.encoder_k(input_ids, attention_mask)
            token_embeddings = output_features[0]       # MoCo 也可以当做Batch的正例;  相当于Dropout + EMA的两者变化; 这里我们仅作为下一个batch的负例
            sentence_emb = _tokens_mean_pooling(token_embeddings, attention_mask, mode='mean')
            # 3.Update Queue
            self._dequeue_and_enqueue(sentence_emb)
        return None


def _tokens_mean_pooling(token_embs=None, att_mask=None, mode='mean'):
    input_mask_expanded = att_mask.unsqueeze(-1).expand(token_embs.size()).float()
    sum_embs = torch.sum(token_embs * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    if mode == 'mean':
        return sum_embs / sum_mask
    if mode == 'sqrt':
        return sum_embs / torch.sqrt(sum_mask)

def _cos_sim(a: Optional[torch.FloatTensor] = None, b: Optional[torch.FloatTensor] = None):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))