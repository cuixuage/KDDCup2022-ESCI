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

        self.brand_vocab_size = 151519
        self.color_vocab_size = 47036
        self.bc_dim = 128
        self.brand_embs = nn.Embedding(self.brand_vocab_size, self.bc_dim)
        self.color_embs = nn.Embedding(self.color_vocab_size, self.bc_dim)
        self.classifier = BrandColorClassificationHead(config, self.bc_dim)    # 2022.05.11 cls + brand_color_embs 分类

        self.emb_bag_size = 732507
        self.emb_bag_dim = 64
        self.char_emb_bag = nn.EmbeddingBag(self.emb_bag_size, self.emb_bag_dim, mode='sum', padding_idx=0)
        self.country_emb = nn.EmbeddingBag(num_embeddings=4, embedding_dim=8, mode='sum', padding_idx=0)

        """
            手动初始化一遍char_emb_bag emb参数; EVAL截断注释掉
        """
        _init_weights(config, self.char_emb_bag)
        _init_weights(config, self.country_emb)
        npy_file = '/home/wangchenyang5/cxg_trial/kdd_cup_2022/data_process/extra_ngram/word2vec.wordvectors.vectors.npy'
        _init_char3_emb_weights(config, self.char_emb_bag, npy_file)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_scores: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        brand_idx: Optional[torch.LongTensor] = None,
        color_idx: Optional[torch.LongTensor] = None,
        sample_weights: Optional[torch.FloatTensor] = None,
        query_id: Optional[torch.LongTensor] = None,
        
        query_chars_input_ids: Optional[torch.LongTensor] = None,
        query_chars_lens: Optional[torch.LongTensor] = None,
        title_bc_chars_input_ids: Optional[torch.LongTensor] = None,
        title_bc_chars_lens: Optional[torch.LongTensor] = None,
        desc_chars_input_ids: Optional[torch.LongTensor] = None,
        desc_chars_lens: Optional[torch.LongTensor] = None,
        country_idx: Optional[torch.LongTensor] = None,
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
        
        sen_q_country = self.country_emb(country_idx.view(-1,1))
        sen_q_char_emb = _chars_mean_pooling(self.char_emb_bag, query_chars_input_ids, query_chars_lens)
        sen_p_t_char_emb = _chars_mean_pooling(self.char_emb_bag, title_bc_chars_input_ids, title_bc_chars_lens)
        sen_p_desc_char_emb = _chars_mean_pooling(self.char_emb_bag, desc_chars_input_ids, desc_chars_lens)
        extend_emb = torch.cat([sen_q_country, sen_q_char_emb, sen_p_t_char_emb, sen_p_desc_char_emb], dim=1)
        logits = self.classifier(sequence_output, labels, brand_idx, color_idx, self.brand_embs, self.color_embs, extend_emb)

        loss = None
        if labels is not None:
            loss = util_loss_fct(logits, label_scores, sample_weights, self.num_labels, loss_name='mse_loss')

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
        self.dense = nn.Linear(config.hidden_size + bc_dim*2 + 64*3 + 8, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # init
        _init_weights(config, self.dense)
        _init_weights(config, self.out_proj)

    def forward(self, sequence_output: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                brand_idx: Optional[torch.LongTensor] = None, color_idx: Optional[torch.LongTensor] = None,
                brand_embs: Optional[torch.FloatTensor] = None, color_embs: Optional[torch.FloatTensor] = None,
                extend_emb: Optional[torch.FloatTensor] = None):
        
        # # 策略1, 直接拼接brandidx, coloridx
        # sentece_emb = sequence_output[:, 0, :]
        # brand_emb = brand_embs(brand_idx)
        # color_emb = color_embs(color_idx)
        # qt_bc_emb = torch.cat((sentece_emb, brand_emb, color_emb), dim=1)
        # qt_bc_emb = self.dropout(qt_bc_emb)
        # qt_bc_emb = self.dense(qt_bc_emb)
        # qt_bc_emb = torch.tanh(qt_bc_emb)
        # qt_bc_emb_d = self.dropout(qt_bc_emb)
        # logits = self.out_proj(qt_bc_emb_d)

        # # 策略2, 删除brandidx, coloridx; 扩展char维度为128
        sentece_emb = sequence_output[:, 0, :]
        brand_emb = brand_embs(brand_idx)
        color_emb = color_embs(color_idx)
        qt_bc_emb = torch.cat((sentece_emb, brand_emb, color_emb, extend_emb), dim=1)
        qt_bc_emb = self.dropout(qt_bc_emb)
        qt_bc_emb = self.dense(qt_bc_emb)
        qt_bc_emb = torch.tanh(qt_bc_emb)
        qt_bc_emb_d = self.dropout(qt_bc_emb)
        logits = self.out_proj(qt_bc_emb_d)

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
    elif isinstance(module, nn.EmbeddingBag):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def _init_char3_emb_weights(config, module, npy_file):
        """
            load weight from local npy;  使用word2vec预训练权重
        """
        if isinstance(module, nn.EmbeddingBag):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            w2v_weight = np.load(npy_file)
            padding_value = np.zeros((1,64))    # padding_value = 0.0
            w2v_weight = torch.FloatTensor(np.vstack((padding_value, w2v_weight)))
            # print("w2v_weight.shape=", w2v_weight.shape)
            # print(w2v_weight[0], w2v_weight[608])   # 608=cat embedding
            module = module.from_pretrained(w2v_weight, freeze=False, padding_idx=0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            # print('module.weight.data=', module.weight.data.shape)
            # print( module.weight[0], module.weight[608])

def _chars_mean_pooling(embed_bag=None, input_feas=None, char_lens=None, mode='mean'):
    batch_size = char_lens.shape[0]
    sum_embs = embed_bag(input_feas.view(batch_size, -1))
    char_lens = torch.clamp(char_lens, min=1.0)
    if mode == 'mean':
        return sum_embs / char_lens.view(-1,1)
    if mode == 'sqrt':
        return sum_embs / torch.sqrt(char_lens.view(-1,1))