''' Define the common attention mechanisms '''

import torch
import torch.nn as nn
import math
from config import Constants
from typing import Dict, Any
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and category embeddings.
    """
    def __init__(self, opt: Dict[str, Any]):
        super(Embeddings, self).__init__()
        if opt.get('pretrained_embs_path', ''):
            # the path to pretrained word embs is specified
            self.word_embeddings = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(np.load(opt['pretrained_embs_path'])).float(),
                freeze=True,
            )
            assert self.word_embeddings.weight.shape[0] == opt['vocab_size']
            dim_word = self.word_embeddings.weight.shape[1]
            if dim_word != opt['dim_hidden']:
                self.w2h = nn.Linear(dim_word, opt['dim_hidden'], bias=False)
        else:
            self.word_embeddings = nn.Embedding(opt['vocab_size'], opt['dim_hidden'], padding_idx=Constants.PAD)
        
        self.trainable_pe = opt.get('trainable_pe', False)
        if self.trainable_pe:
            self.position_embeddings = nn.Embedding(opt['max_len'], opt['dim_hidden'])
        else:
            self.position_embeddings = PositionalEmbedding(opt['max_len'], opt['dim_hidden'])

        self.with_category = opt.get('with_category', False)
        self.use_category_embs = opt.get('use_category_embs', False)
        if self.with_category:
            if self.use_category_embs:
                self.category_embeddings = nn.Linear(opt['dim_category'], opt['dim_hidden'])
            else:
                self.category_embeddings = nn.Embedding(opt['num_category'], opt['dim_hidden'])
        
        if not opt.get('transformer_pre_ln', False):
            self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def forward(self, input_ids, category=None, additional_feats=None, **kwargs):
        words_embeddings = self.word_embeddings(input_ids)
        if hasattr(self, 'w2h'):
            words_embeddings = self.w2h(words_embeddings)

        if self.trainable_pe:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(input_ids)

        embeddings = words_embeddings + position_embeddings
        
        if self.with_category:
            inputs = kwargs['category_embs'] if self.use_category_embs else category
            assert inputs is not None

            category_embeddings = self.category_embeddings(inputs.to(input_ids.device))
            if category_embeddings.dim() == 2:
                category_embeddings = category_embeddings.unsqueeze(1) # [bsz, 1, dim_hidden]

            embeddings = embeddings + category_embeddings
        
        if additional_feats is not None:
            embeddings = embeddings + additional_feats
        
        if hasattr(self, 'LayerNorm'):
            embeddings = self.LayerNorm(embeddings)
        
        embeddings = self.dropout(embeddings)
        return embeddings
