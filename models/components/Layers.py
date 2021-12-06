''' Define the layers in Transformer'''

import torch
import torch.nn as nn
from models.components.SubLayers import (
    MultiHeadAttention, 
    PositionwiseFeedForward
)
from typing import Dict, Any, Optional, Tuple


class EncoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(EncoderLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
        )
        self.ffn = PositionwiseFeedForward(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            pre_ln=opt.get('transformer_pre_ln', False),
        )

    def forward(self, 
            hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, attention_probs, context = self.intra_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states, attention_probs, context


class DecoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(DecoderLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
        )

        if opt.get('fusion', 'temporal_concat') == 'channel_concat':
            dim_key = dim_value = opt['dim_hidden'] * len(opt['modality'])
        else:
            dim_key = dim_value = opt['dim_hidden']

        self.inter_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            dim_key=dim_key,
            dim_value=dim_value,
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
        )

        self.ffn = PositionwiseFeedForward(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            pre_ln=opt.get('transformer_pre_ln', False),
        )

    def forward(self, 
            hidden_states: torch.Tensor, 
            encoder_hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            encoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        hidden_states, intra_attention_probs, text_context = self.intra_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        self_embs = hidden_states.clone()

        hidden_states, inter_attention_probs, context = self.inter_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        cross_embs = hidden_states.clone()

        hidden_states = self.ffn(hidden_states)        
        return hidden_states, (intra_attention_probs, inter_attention_probs), (text_context, context), (self_embs, cross_embs)
