import math
from typing import List, Optional, Tuple, Union
from utils import util_loss_fct, cos_sim
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaClassificationHead

class RobertaWithSampleWeight(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    """
        Qmix = Q_char3_mean + Q
        Tmix = T_B_C_Bullet_Desc_char3_mean + (T_B_C,Bullet,Desc)
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.classifier = RobertaClassificationHead(config)     # 2022.05.11 cls分类

        # self.brand_vocab_size = 151519
        # self.color_vocab_size = 47036
        # self.bc_dim = 128
        # self.brand_embs = nn.Embedding(self.brand_vocab_size, self.bc_dim)
        # self.color_embs = nn.Embedding(self.color_vocab_size, self.bc_dim)
        # self.brand_head = BrandColorHead(config, self.brand_embs, self.brand_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个品牌最相关
        # self.color_head = BrandColorHead(config, self.color_embs, self.color_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个颜色、型号最相关

        self.classifier = BrandColorClassificationHead(config)    # 2022.05.11 cls + brand_color_embs 分类

        self.emb_bag_size = 754656
        self.emb_bag_dim = 256
        self.char_emb_bag = nn.EmbeddingBag(self.emb_bag_size, self.emb_bag_dim, mode='sum', padding_idx=0)
        self.country_emb = nn.EmbeddingBag(4, config.hidden_size, mode='sum', padding_idx=0)
        # self.query_weighted_sum = WeightedSumRes(config, embedding_dim=config.hidden_size, channel_num=4)
        # self.product_weighted_sum = WeightedSumRes(config, embedding_dim=config.hidden_size, channel_num=4)

        # Initialize weights and apply final processing
        _init_weights(config, self.char_emb_bag)
        _init_weights(config, self.country_emb)
        self.post_init()
        """
            以防万一, 再手动初始化一遍brand\color emb参数
        """
        # self.brand_embs.weight = self.brand_head.embed_layer.weight
        # self.color_embs.weight = self.color_head.embed_layer.weight

    def forward(
        self,
        query_input_ids: Optional[torch.LongTensor] = None,
        query_attention_mask: Optional[torch.FloatTensor] = None,
        title_bc_input_ids: Optional[torch.LongTensor] = None,
        title_bc_attention_mask: Optional[torch.FloatTensor] = None,
        desc_input_ids: Optional[torch.LongTensor] = None,
        desc_attention_mask: Optional[torch.FloatTensor] = None,

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

        query_chars_input_ids: Optional[torch.LongTensor] = None,
        query_chars_lens: Optional[torch.LongTensor] = None,
        title_bc_chars_input_ids: Optional[torch.LongTensor] = None,
        title_bc_chars_lens: Optional[torch.LongTensor] = None,
        desc_chars_input_ids: Optional[torch.LongTensor] = None,
        desc_chars_lens: Optional[torch.LongTensor] = None,
        country_idx: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        # 1.query-tokens-emb
        q_outputs = self.roberta(
            query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sen_q_xlm_emb = (_tokens_mean_pooling(q_outputs.last_hidden_state, query_attention_mask, 'mean') + _tokens_mean_pooling(q_outputs.hidden_states[1] , query_attention_mask, 'mean') ) /2.0
        # sen_q_xlm_emb = _tokens_mean_pooling(q_outputs.last_hidden_state, query_attention_mask, 'mean')
        sen_q_char_emb = _chars_mean_pooling(self.char_emb_bag, query_chars_input_ids, query_chars_lens)
        sen_q_country = self.country_emb(country_idx.view(-1,1))

        # 2.product-tokens-emb
        # 2.1 XLM EMb
        p_t_bc_outputs = self.roberta(
            title_bc_input_ids,
            attention_mask=title_bc_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sen_p_t_emb = (_tokens_mean_pooling(p_t_bc_outputs.last_hidden_state, title_bc_attention_mask, 'mean') + _tokens_mean_pooling(p_t_bc_outputs.hidden_states[1], title_bc_attention_mask, 'mean') ) / 2.0
        # sen_p_t_emb = _tokens_mean_pooling(p_t_bc_outputs.last_hidden_state, title_bc_attention_mask, 'mean')
        p_desc_outputs = self.roberta(
            desc_input_ids,
            attention_mask=desc_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sen_p_desc_emb = (_tokens_mean_pooling(p_desc_outputs.last_hidden_state, desc_attention_mask, 'mean') + _tokens_mean_pooling(p_desc_outputs.hidden_states[1], desc_attention_mask, 'mean') ) / 2.0
        # sen_p_desc_emb = _tokens_mean_pooling(p_desc_outputs.last_hidden_state, desc_attention_mask, 'mean')
        # 2.2 chars embedding bag
        sen_p_t_char_emb = _chars_mean_pooling(self.char_emb_bag, title_bc_chars_input_ids, title_bc_chars_lens)
        sen_p_desc_char_emb = _chars_mean_pooling(self.char_emb_bag, desc_chars_input_ids, desc_chars_lens)

        # 3. cross-attention emb
        q_cross, p_cross = _cross_attention(q_outputs[0], query_attention_mask, p_t_bc_outputs[0], title_bc_attention_mask)
        q_cross_emb = _tokens_mean_pooling(q_cross, query_attention_mask, 'mean')
        p_cross_emb = _tokens_mean_pooling(p_cross, title_bc_attention_mask, 'mean')
        q_2_cross, p_2_cross = _cross_attention(q_outputs[0], query_attention_mask, p_desc_outputs[0], desc_attention_mask)
        q_2_cross_emb = _tokens_mean_pooling(q_2_cross, query_attention_mask, 'mean')
        p_2_cross_emb = _tokens_mean_pooling(p_2_cross, desc_attention_mask, 'mean')

        # 4. sentece-transformer, softmax_crossentropy_loss
        # q_emb = self.query_weighted_sum([sen_q_xlm_emb, q_cross_emb, q_2_cross_emb, sen_q_country])
        # p_emb = self.product_weighted_sum([sen_p_t_emb , sen_p_desc_emb , p_cross_emb, p_2_cross_emb])
        q_emb = (sen_q_xlm_emb + q_cross_emb + q_2_cross_emb + sen_q_country) / 4.0
        p_emb = (sen_p_t_emb + sen_p_desc_emb + p_cross_emb + p_2_cross_emb) / 4.0

        feature_emb = torch.cat([q_emb, p_emb, torch.abs(q_emb - p_emb), sen_q_char_emb, sen_p_t_char_emb, sen_p_desc_char_emb], 1)
        sequence_output = feature_emb
        # print('sequence_output', sequence_output.shape, sen_q_char_emb.shape, sen_p_t_char_emb.shape, sen_p_desc_char_emb.shape)

        # logits = self.classifier(sequence_output, brand_idx, color_idx, self.brand_embs, self.color_embs)
        logits = self.classifier(sequence_output, brand_idx, color_idx, None, None)

        loss = None
        if labels is not None:
            loss = util_loss_fct(logits, labels, sample_weights, self.num_labels, loss_name='ce-poly-1')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class BrandColorClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size+bc_dim*2, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size*3 + 256*3, config.hidden_size)      # 双塔分类, [u,v,|u-v|]。 MLP网络参数尽可能少一些,避免显存占用过多。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        _init_weights(config, self.out_proj)

    def forward(self, sequence_output: Optional[torch.FloatTensor] = None,
                brand_idx: Optional[torch.LongTensor] = None, color_idx: Optional[torch.LongTensor] = None,
                brand_embs: Optional[torch.FloatTensor] = None, color_embs: Optional[torch.FloatTensor] = None):
        
        # sentece_emb = sequence_output[:, 0, :]
        # sentece_emb = sequence_output
        # brand_emb = brand_embs(brand_idx)
        # color_emb = color_embs(color_idx)
        # qt_bc_emb = torch.cat((sentece_emb, brand_emb, color_emb), dim=1)
        # qt_bc_emb = sentece_emb
        qt_bc_emb = self.dropout(sequence_output)
        qt_bc_emb = self.dense(qt_bc_emb)
        qt_bc_emb = torch.tanh(qt_bc_emb)
        qt_bc_emb = self.dropout(qt_bc_emb)
        logits = self.out_proj(qt_bc_emb)
        return logits


class WeightedSumRes(nn.Module):
    def __init__(self, config, embedding_dim, channel_num):
        super().__init__()
        self.out_proj = nn.Linear(channel_num * embedding_dim, channel_num)      # weighted sum, [N*D, N]
        _init_weights(config, self.out_proj)

    def forward(self, emb_list: Optional[list] = None):
        stack_emb = torch.stack(emb_list, dim=1)
        mix_emb = torch.cat(emb_list, dim=-1)
        weights = torch.softmax(self.out_proj(mix_emb), -1)
        sum_emb = torch.sum(stack_emb * weights.unsqueeze(-1), dim=1)
        return sum_emb

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
        _init_weights(config, self.dense)

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

def _tokens_mean_pooling(token_embs=None, att_mask=None, mode='mean'):
    input_mask_expanded = att_mask.unsqueeze(-1).expand(token_embs.size()).float()
    sum_embs = torch.sum(token_embs * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    if mode == 'mean':
        return sum_embs / sum_mask
    if mode == 'sqrt':
        return sum_embs / torch.sqrt(sum_mask)

def _chars_mean_pooling(embed_bag=None, input_feas=None, char_lens=None, mode='mean'):
    batch_size = char_lens.shape[0]
    sum_embs = embed_bag(input_feas.view(batch_size, -1))
    char_lens = torch.clamp(char_lens, min=1.0)
    if mode == 'mean':
        return sum_embs / char_lens.view(-1,1)
    if mode == 'sqrt':
        return sum_embs / torch.sqrt(char_lens.view(-1,1))

def _cross_attention(Q_emb: Optional[torch.FloatTensor] = None, Q_mask: Optional[torch.FloatTensor] = None,
                    P_emb: Optional[torch.FloatTensor] = None, P_mask: Optional[torch.FloatTensor] = None):
    """
        1. cross-attention, QKV矩阵
            Q矩阵对于premise做映射，KV矩阵对于premise做映射。再通过softmax(Q*K)作为attention权重。
        2. ESIM
            直接使用softmax(*)

        这里主要是考虑到显存占用限制, 采用方式2
    """
    attention = torch.matmul(Q_emb, P_emb.transpose(1,2))
    Q_mask = (1.0 - Q_mask).float() * -10000.0
    P_mask = (1.0 - P_mask).float() * -10000.0
    # print(attention.shape, Q_mask.unsqueeze(1).shape, P_mask.unsqueeze(1).shape)
    Q_prob = F.softmax(attention + P_mask.unsqueeze(1), dim=-1)
    P_prob = F.softmax(attention.transpose(1,2) + Q_mask.unsqueeze(1), dim=-1)
    # print(attention.shape, Q_mask.unsqueeze(1).shape, P_mask.unsqueeze(1).shape, Q_prob.shape, P_prob.shape)
    Q_cross = torch.matmul(Q_prob, P_emb)
    P_cross = torch.matmul(P_prob, Q_emb) 
    return Q_cross, P_cross