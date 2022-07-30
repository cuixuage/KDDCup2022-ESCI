import math
from typing import List, Optional, Tuple, Union
from utils import util_loss_fct
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
import numpy as np
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaClassificationHead

class RobertaWithSampleWeight(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # self.brand_vocab_size = 151519
        # self.color_vocab_size = 47036
        # self.bc_dim = 128
        # self.brand_embs = nn.Embedding(self.brand_vocab_size, self.bc_dim)
        # self.color_embs = nn.Embedding(self.color_vocab_size, self.bc_dim)

        self.classifier = BrandColorClassificationHead_MOE(config, 3)    # MOE  3-experts

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        qt_input_ids: Optional[torch.LongTensor] = None,
        qt_attention_mask: Optional[torch.FloatTensor] = None,
        qb_input_ids: Optional[torch.LongTensor] = None,
        qb_attention_mask: Optional[torch.FloatTensor] = None,
        qd_input_ids: Optional[torch.LongTensor] = None,
        qd_attention_mask: Optional[torch.FloatTensor] = None,
        task_masks: Optional[torch.LongTensor] = None,

        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        brand_idx: Optional[torch.LongTensor] = None,
        color_idx: Optional[torch.LongTensor] = None,
        sample_weights: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        qt_outputs = self.roberta(
            qt_input_ids,
            attention_mask=qt_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        qt_emb = qt_outputs[0][:, 0, :]
        qb_outputs = self.roberta(
            qb_input_ids,
            attention_mask=qb_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        qb_emb = qb_outputs[0][:, 0, :]
        qd_outputs = self.roberta(
            qd_input_ids,
            attention_mask=qd_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        qd_emb = qd_outputs[0][:, 0, :]

        probs = self.classifier(qt_emb, qb_emb, qd_emb, task_masks)

        loss = None
        if labels is not None:
            loss = util_loss_fct(probs, labels, sample_weights, self.num_labels, loss_name='ce-poly-1-in_prob')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=probs,               # 2022.06.04   后续使用argmax则不影响结果
            hidden_states=None,
            attentions=None,
        )

class BrandColorClassificationHead(nn.Module):
    def __init__(self, config, bc_dim):
        super().__init__()
        self.num_labels = config.num_labels
        self.dense = nn.Linear(config.hidden_size + bc_dim*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # init
        _init_weights(config, self.dense)
        _init_weights(config, self.out_proj)

    def forward(self, sequence_output: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                brand_idx: Optional[torch.LongTensor] = None, color_idx: Optional[torch.LongTensor] = None,
                brand_embs: Optional[torch.FloatTensor] = None, color_embs: Optional[torch.FloatTensor] = None):
        
        # 策略1, 直接拼接brandidx, coloridx
        sentece_emb = sequence_output[:, 0, :]
        brand_emb = brand_embs(brand_idx)
        color_emb = color_embs(color_idx)
        qt_bc_emb = torch.cat((sentece_emb, brand_emb, color_emb), dim=1)
        qt_bc_emb = self.dropout(qt_bc_emb)
        qt_bc_emb = self.dense(qt_bc_emb)
        qt_bc_emb = torch.tanh(qt_bc_emb)
        qt_bc_emb_d = self.dropout(qt_bc_emb)
        logits = self.out_proj(qt_bc_emb_d)

        # # 策略2, DNN, 删除dropout, 最后拼接brandidx, coloridx
        # sentece_emb = sequence_output[:, 0, :]
        # brand_emb = brand_embs(brand_idx)
        # color_emb = color_embs(color_idx)
        # sentece_emb = self.dense(sentece_emb)
        # sentece_emb = torch.tanh(sentece_emb)
        # qt_bc_emb = torch.cat((sentece_emb, brand_emb, color_emb), dim=1)
        # logits = self.out_proj(qt_bc_emb)

        # Embedding  MixUp   (相当于跨语言的Embedding)
        alpha = 0.2
        x = qt_bc_emb
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        ont_hot_labels = F.one_hot(labels, num_classes=self.num_labels)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * ont_hot_labels + (1 - lam) * ont_hot_labels[index]
        mixed_logits = self.out_proj(mixed_x)

        return logits, mixed_logits, mixed_y

class ClassificationHeadRefactor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # init
        _init_weights(config, self.dense)
        _init_weights(config, self.out_proj)

    def forward(self, sentece_emb):
        x = self.dropout(sentece_emb)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BrandColorClassificationHead_MOE(nn.Module):
    def __init__(self, config, task_num):
        super().__init__()
        self.num_labels = config.num_labels
        self.task_num = task_num    # 2022.06.04 , 三个匹配任务  q-title, q-bullet, q-desc   (MOE=one-gate, MMOE=multi-gates)
        self.gating_net = nn.Linear(config.hidden_size * self.task_num, self.task_num)
        # self.shared_expert = nn.ModuleList([ ClassificationHeadRefactor(config) for i in range(self.task_num) ])
        self.shared_expert = ClassificationHeadRefactor(config)
        # init
        _init_weights(config, self.gating_net)

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums
    
    def forward(self, qt_emb: Optional[torch.FloatTensor] = None, qb_emb: Optional[torch.FloatTensor] = None,
                qd_emb: Optional[torch.FloatTensor] = None, task_masks: Optional[torch.LongTensor] = None):
        """
            Gating-Network = [task_num * 768, task_num]
            Task-Mask = [bs,task_num]
        """
        # 1.通过gating-net获取prob, 注意是masked-softmax
        input_emb = torch.cat([qt_emb, qb_emb, qd_emb], dim=1)
        weighted_logits = self.gating_net(input_emb)
        masked_prob = self.masked_softmax(weighted_logits, task_masks)

        # 2.分别做Expert计算, 得到非线性变换得到probs
        qt_probs = torch.softmax(self.shared_expert(qt_emb), dim=1)
        qb_probs = torch.softmax(self.shared_expert(qb_emb), dim=1)
        qd_probs = torch.softmax(self.shared_expert(qd_emb), dim=1)

        # 3.计算最终的分类probs
        probs = torch.stack([qt_probs, qb_probs, qd_probs], dim=1)
        probs = torch.sum(probs * masked_prob.unsqueeze(-1), dim=1)
        return probs

class BrandColorHead(nn.Module):
    """brand  color classifier"""

    def __init__(self, config, embed_layer, vocab_size, emb_dim):
        super().__init__()
        # 1.brand\color embedding layer
        self.embed_layer = embed_layer
        # 2. CLS Emb 非线性变换
        self.dense = nn.Linear(config.hidden_size, emb_dim)
        self.dense_layernorm = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)
        # 3. similarity bias
        self.sim_bias = nn.Parameter(torch.zeros(vocab_size))
        # 4.init
        _init_weights(config, self.embed_layer)
        _init_weights(config, self.dense)
        _init_weights(config, self.dense_layernorm)

    def forward(self, features: Optional[torch.FloatTensor] = None):
        sentece_emb = features[:, 0, :]   # take <s> token (equiv. to [CLS])
        x = self.dense(sentece_emb)
        x = gelu(x)
        x = self.dense_layernorm(x)
        sim = torch.matmul(x, self.embed_layer.weight.T) + self.sim_bias
        # print('embedding.weights=', self.embed_layer.weight.shape, sim.shape)
        return sim

# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
def _init_weights(config, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)