from transformers import PreTrainedModel, MT5EncoderModel
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Dict
from SMT5 import SMT5Config

class SMT5PreTrainedModel(PreTrainedModel):
    config_class = SMT5Config
    base_model_prefix = "smt5"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SMT5MapperModel(SMT5PreTrainedModel):
    def __init__(self, config: SMT5Config):
        super().__init__(config)
        self.config = config

        self.proj = nn.Linear(self.config.d_out, self.config.d_proj)

    def forward(self, input_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        # mean pooling
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(input_embeddings.size()).float()
        sum_embeddings = torch.sum(input_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        sentence_vector = torch.cat(output_vectors, 1)

        x = self.proj(sentence_vector)
        x = F.normalize(x, p=2.0)
        return x

class SMT5CLModel(nn.Module):
    def __init__(self, ckpt: str, mapper_ckpt: str = '', mode: str="train"):
        super(SMT5CLModel, self).__init__()
        if mode == "train":
            config = SMT5Config()
            self.mapper = SMT5MapperModel(config)
        else:
            self.mapper = SMT5MapperModel.from_pretrained(mapper_ckpt)

        self.encoder = MT5EncoderModel.from_pretrained(ckpt)

    def save_pretrained(self, path):
        self.mapper.save_pretrained(path + "/mapper")
        self.encoder.save_pretrained(path + "/encoder")

    def _cross_equal(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        bz = inp1.shape[0]
        inp1_new = inp1.view(bz, -1).repeat_interleave(bz, dim=-1)
        inp2_new = inp2.repeat(bz).view(bz, -1)
        return (inp1_new == inp2_new).float()

    def forward(self, inputs):
        neg_out = None
        if 'query_input_ids' in inputs:
            query_outputs = self.encoder(
                input_ids=inputs['query_input_ids'],
                attention_mask=inputs['query_attention_mask'],
                return_dict=True,
            )
            anchor_out = self.mapper(query_outputs.last_hidden_state, inputs['query_attention_mask'])

            answer_outputs = self.encoder(
                input_ids=inputs['answer_input_ids'],
                attention_mask=inputs['answer_attention_mask'],
                return_dict=True,
            )
            pos_out = self.mapper(answer_outputs.last_hidden_state, inputs['answer_attention_mask'])
        elif 'anchor_input_ids' in inputs:
            anchor_outputs = self.encoder(
                input_ids=inputs['anchor_input_ids'],
                attention_mask=inputs['anchor_attention_mask'],
                return_dict=True,
            )
            anchor_out = self.mapper(anchor_outputs.last_hidden_state, inputs['anchor_attention_mask'])

            pos_outputs = self.encoder(
                input_ids=inputs['pos_input_ids'],
                attention_mask=inputs['pos_attention_mask'],
                return_dict=True,
            )
            pos_out = self.mapper(pos_outputs.last_hidden_state, inputs['pos_attention_mask'])

            neg_outputs = self.encoder(
                input_ids=inputs['neg_input_ids'],
                attention_mask=inputs['neg_attention_mask'],
                return_dict=True,
            )
            neg_out = self.mapper(neg_outputs.last_hidden_state, inputs['neg_attention_mask'])
        
        device = anchor_out.device 
        bz, _ = anchor_out.shape
        sim = torch.exp(torch.div(torch.inner(anchor_out, pos_out), self.mapper.config.temp))
        cross_lingual_idx = self._cross_equal(inputs['idx'], inputs['idx']).to(device)
        self_pos_mask = torch.zeros(bz, bz).fill_diagonal_(1).to(device)
        in_batch_neg_mask = torch.ones(bz, bz).to(device) - cross_lingual_idx
        self_pos = (sim * self_pos_mask).sum(dim=-1, keepdim=True)
        ib_neg = (sim * in_batch_neg_mask).sum(dim=-1, keepdim=True)
        
        if neg_out is not None:
            sim_neg = torch.exp(torch.inner(anchor_out, neg_out) / self.mapper.config.temp).sum(dim=-1, keepdim=True)
            ib_neg = ib_neg + sim_neg

        loss = self_pos / ib_neg
        return loss

        
