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
        # self.classifier = RobertaClassificationHead(config)     # 2022.05.11 cls分类

        self.brand_vocab_size = 151519
        self.color_vocab_size = 47036
        self.bc_dim = 128
        self.brand_embs = nn.Embedding(self.brand_vocab_size, self.bc_dim)
        self.color_embs = nn.Embedding(self.color_vocab_size, self.bc_dim)
        # self.brand_head = BrandColorHead(config, self.brand_embs, self.brand_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个品牌最相关
        # self.color_head = BrandColorHead(config, self.color_embs, self.color_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个颜色、型号最相关

        self.classifier = BrandColorClassificationHead(config, self.bc_dim)    # 2022.05.11 cls + brand_color_embs 分类

        # Initialize weights and apply final processing
        self.post_init()
        """
            以防万一, 再手动初始化一遍brand\color emb参数
        """
        # self.brand_embs.weight = self.brand_head.embed_layer.weight
        # self.color_embs.weight = self.color_head.embed_layer.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
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

        output_hidden_states = True
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(outputs)
        # sequence_output_24th = outputs.last_hidden_state
        # sequence_output_12th = outputs.hidden_states[12]              # 2022.06.05  第0层是embedding
        # sequence_output = (sequence_output_24th + sequence_output_12th) / 2.0
        sequence_output = outputs.last_hidden_state
        
        # logits, mixed_logits, mixed_labels = self.classifier(sequence_output, labels, brand_idx, color_idx, self.brand_embs, self.color_embs)
        logits = self.classifier(sequence_output, labels, brand_idx, color_idx, self.brand_embs, self.color_embs)

        loss = None
        # if labels is not None:
        #     origin_loss = util_loss_fct(logits, labels, sample_weights, self.num_labels, loss_name='ce-poly-1')
        #     mixed_loss = util_loss_fct(mixed_logits, mixed_labels, sample_weights, self.num_labels, loss_name='ce-soft-label')
        #     loss = (origin_loss + mixed_loss) / 2.0

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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

        # # 策略3, Embedding  MixUp   (相当于跨语言的Embedding)
        # alpha = 0.2
        # x = qt_bc_emb
        # lam = np.random.beta(alpha, alpha)
        # batch_size = x.size()[0]
        # index = torch.randperm(batch_size).cuda()
        # ont_hot_labels = F.one_hot(labels, num_classes=self.num_labels)
        # mixed_x = lam * x + (1 - lam) * x[index]
        # mixed_y = lam * ont_hot_labels + (1 - lam) * ont_hot_labels[index]
        # mixed_logits = self.out_proj(mixed_x)

        # return logits, mixed_logits, mixed_y
        return logits

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
