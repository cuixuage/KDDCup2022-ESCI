import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from utils import _tokens_mean_pooling, _cos_sim

import torch
import torch.utils.checkpoint
from packaging import version
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

class MultiTaskPretrainedModel(RobertaPreTrainedModel):
    """
        Code Ref:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1030
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias", r"fq_head.dense.weight", r"fq_head.dense.bias", r"encoder_k_model.*"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias", r"fq_head.dense.weight", r"fq_head.dense.bias"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"fq_head.dense.weight", r"fq_head.dense.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.roberta = RobertaModel.from_pretrained(input_model_dir, config=config)    # 2022.04.27 会导致lm_head.encoder参数未能初始化
        self.fq_head = RobertaClassificationHead(config)                                   # 2022.04.29 加上这个Head, 直接做Eval导致lm_head Loss升高+0.03  原因未知
        self.lm_head = RobertaLMHead(config)
        self.brand_vocab_size = 151519
        self.color_vocab_size = 47036
        self.bc_dim = 128
        self.brand_embs = nn.Embedding(self.brand_vocab_size, self.bc_dim)
        self.color_embs = nn.Embedding(self.color_vocab_size, self.bc_dim)
        _init_weights(config, self.brand_embs)
        _init_weights(config, self.color_embs)
        self.brand_head = BrandColorHead(config, self.brand_embs, self.brand_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个品牌最相关
        self.color_head = BrandColorHead(config, self.color_embs, self.color_vocab_size, self.bc_dim)      # 2022.05.06 预测sku_embedding和哪个颜色、型号最相关

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()
        self.num_labels = 2

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        brand_idx_labels: Optional[torch.LongTensor] = None,
        color_idx_labels:Optional[torch.LongTensor] = None,
        special_tokens_mask: Optional[torch.FloatTensor] = None,
        encoder_k_model=None,      # only for Contrasitve Task
        task_name: Optional[str] = 'MLM', 
    ):
        special_tokens_mask = None
        # 1.shared Encoder
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict       # return_dict = True
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 2.multi-task pretraning
        if task_name == 'MLM':      
            sequence_output = outputs[0]  
            prediction_scores = self.lm_head(sequence_output)  # 2022.05.06 lm_head实现时新建了vocab_size的分类器, 类似于word_2_vec的输出层,输入输出泾渭分明。而Bert Tensorflow代码则是直接对于input_embedding做分类
            masked_lm_loss = None

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif task_name == 'FakedQuery':
            sequence_output = outputs[0]  
            faked_query_loss = None
            logits = self.fq_head(sequence_output)
            if labels is not None:
                # loss_fct = CrossEntropyLoss()
                # faked_query_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                epsilon = 1.0
                loss_fct = CrossEntropyLoss(reduction='none')
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                pt = torch.softmax(logits, dim=1).index_select(-1, labels)   # 方式3
                poly1_loss = epsilon*(1-pt) + ce_loss
                faked_query_loss = torch.mean(poly1_loss)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((faked_query_loss,) + output) if faked_query_loss is not None else output

            return SequenceClassifierOutput(
                loss=faked_query_loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif task_name == 'BrandColor':
            sequence_output = outputs[0] 
            # print('sequence_output', sequence_output.shape)
            brand_predict_loss = 0
            color_predict_loss = 0
            if brand_idx_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                brand_sim = self.brand_head (sequence_output)
                brand_predict_loss = loss_fct(brand_sim.view(-1, self.brand_vocab_size), brand_idx_labels.view(-1))
            if color_idx_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                color_sim = self.color_head(sequence_output)
                color_predict_loss = loss_fct(color_sim.view(-1, self.color_vocab_size), color_idx_labels.view(-1))
            total_loss = (brand_predict_loss + color_predict_loss) / 2     
            return BrandColorOutput(loss=total_loss, brand_loss=brand_predict_loss, color_loss=color_predict_loss)
        elif task_name == 'Contrasitive':
            """
                预训练这里用MoCo扩充batch队列了, 也使用SimCse
            """
            outputs_dropout = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # 1.simcse
            sen_q = _tokens_mean_pooling(outputs[0], attention_mask)
            sen_r = _tokens_mean_pooling(outputs_dropout[0], attention_mask)
            logits_simcse = _cos_sim(sen_q, sen_r)
            # 2.Moco
            queue_embs = encoder_k_model.queue.clone().detach()
            logits_neg = _cos_sim(sen_q, queue_embs)
            # 3.labels\predictions
            labels = torch.tensor(range(len(logits_simcse)), dtype=torch.long, device=logits_simcse.device)
            scores = torch.cat([logits_simcse, logits_neg], dim=1)  * 33.0   # temperature = 0.03
            # 4. Updata Queue
            encoder_k_model(input_ids=input_ids, attention_mask=attention_mask)
            # 5.Loss
            loss_fct = CrossEntropyLoss()
            contrasitive_loss = loss_fct(scores.view(len(scores), -1), labels.view(-1))
            return SequenceClassifierOutput(
                loss=contrasitive_loss,
                logits=scores,
                hidden_states=None,
                attentions=None,
            )
        else:
            return None


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

@dataclass
class BrandColorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    brand_loss: Optional[torch.FloatTensor] = None
    color_loss: Optional[torch.FloatTensor] = None


