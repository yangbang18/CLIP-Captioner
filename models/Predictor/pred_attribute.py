import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Union


def get_prj_by_flag(opt, prj: Union[nn.ModuleList, nn.Module], flag: Optional[str]=None) -> nn.Module:
    if isinstance(prj, nn.ModuleList):
        assert flag is not None
        return prj[opt['attribute_prediction_flags'].index(flag)]
    return prj


def prepare_merged_probs(
        scores: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_avg_prob: bool = False
    ):
    assert len(scores.shape) == 3, "[bsz, n_frames, n_attributes]"
    probs = torch.sigmoid(scores)
    raw = torch.log(torch.clamp(1.0 - probs, 1e-12, 1)) # avoid nan

    if mask is not None:
        # if a position's mask is True, then its prob is 0, so log(1 - prob) is 0 too
        mask = mask.to(scores.device)
        raw = raw.masked_fill(mask.unsqueeze(2).expand_as(raw), 0)

        denominator = (~mask).sum(dim=1).float()
        denominator = torch.where(
            denominator > 0, 
            denominator, 
            torch.ones_like(denominator).to(denominator.device)
        ) # avoid nan
        
        avg_prob = probs.mean(dim=2) # (bsz, seq_len)
        avg_prob = torch.sum(avg_prob * (~mask).float(), dim=1) / denominator # (bsz, )
    else:
        avg_prob = probs.mean(dim=(1, 2)) # (bsz, )

    merge = raw.sum(dim=1)
    outputs =  1.0 - torch.exp(merge)

    return (outputs, avg_prob) if return_avg_prob else outputs


class Predictor_attribute(nn.Module):
    def __init__(self, opt):
        super(Predictor_attribute, self).__init__()
        self.opt = opt 
        self.flags = opt['attribute_prediction_flags']
        self.sparse_sampling = opt.get('attribute_prediction_sparse_sampling', False)
        
        if opt.get('attribute_prediction_share_prj', False) or len(self.flags) == 1:
            self.prj = nn.Linear(opt['dim_hidden'], opt['attribute_prediction_k'])
        else:
            self.prj = nn.ModuleList([
                nn.Linear(opt['dim_hidden'], opt['attribute_prediction_k']) 
                for _ in range(len(self.flags))
            ])
    
    def get_topk_attribute_predictions(self, feats, mask=None, topk=100, flag=None):
        scores = get_prj_by_flag(self.opt, self.prj, flag)(feats)
        preds_attr = prepare_merged_probs(scores, mask=mask, return_avg_prob=False) # [bsz, n_attributes]
        topk_probs, topk_indices = preds_attr.topk(topk, dim=-1, largest=True, sorted=True)
        return topk_probs, topk_indices

    def forward(self, encoder_hidden_states, **kwargs):
        '''
            encoder_hidden_states: [bsz, n_frames * n_modality, d] when `fusion` == 'temporal_concat'
                                   list of [bsz, n_frames, d] when `fusion` == 'none'
        '''
        if isinstance(encoder_hidden_states, list):
            query = torch.cat(encoder_hidden_states, dim=1) # [bsz, n_frames * n_modality, dim_hidden]
        else:
            query = encoder_hidden_states # [bsz, n_frames * n_modality, dim_hidden]
        
        V_mask = None
        if self.training and self.sparse_sampling:
            assert 'V' in self.flags
            bsz, seq_len = query.shape[:2]
            all_ids = [_ for _ in range(seq_len)]
            sampled_query = []

            V_mask = query.new(bsz, seq_len).fill_(1)
            for i in range(bsz):
                sparsely_sampling_ratio = np.random.rand()
                sparsely_sampling_num = math.ceil(seq_len * sparsely_sampling_ratio)

                sampled_ids = np.random.choice(all_ids, sparsely_sampling_num, replace=False)
                sampled_ids = list(sampled_ids) + [0] * (seq_len - sparsely_sampling_num) # padding
                
                sampled_query.append(query[i, sampled_ids])
                V_mask[i, :sparsely_sampling_num] = 0
                
            query = torch.stack(sampled_query, dim=0) # [bsz, seq_len, dim_hidden]
            V_mask = V_mask.bool()
            assert query.shape[1] == seq_len
        
        if 'V' not in self.flags:
            preds_attr, avg_prob_attr = None, None
        else:
            scores = get_prj_by_flag(self.opt, self.prj, flag='V')(query)
            preds_attr, avg_prob_attr = prepare_merged_probs(scores, return_avg_prob=True, mask=V_mask) # [bsz, n_attributes]

        return {
            'preds_attr': preds_attr, 
            'avg_prob_attr': avg_prob_attr, 
            'attribute_prediction_prj': self.prj
        }

    @staticmethod
    def add_specific_args(parent_parser: object) -> object:
        parser = parent_parser.add_argument_group(title='Attribute Prediction Settings')
        parser.add_argument('-ap', '--attribute_prediction', default=False, action='store_true')
        parser.add_argument('-ap_k', '--attribute_prediction_k', type=int, default=500)

        parser.add_argument('-ap_flags', '--attribute_prediction_flags', type=str, default='VI')
        parser.add_argument('-ap_scales', '--attribute_prediction_scales', type=float, nargs='+', default=[1.0])

        parser.add_argument('-ap_ss', '--attribute_prediction_sparse_sampling', default=False, action='store_true')
        parser.add_argument('-ap_sp', '--attribute_prediction_share_prj', default=False, action='store_true')
        
        parser.add_argument('--TAP_pos', default=False, action='store_true')
        parser.add_argument('--TAP_ln', default=False, action='store_true')
        return parent_parser
    
    @staticmethod
    def check_args(args: object) -> None:
        if args.attribute_prediction:
            if not isinstance(args.crits, list):
                args.crits = [args.crits]
            if 'attribute' not in args.crits:
                args.crits.append('attribute')


class TextPostProcesser(nn.Module):
    def __init__(self, opt):
        super(TextPostProcesser, self).__init__()
        if opt.get('TAP_pos', False):
            self.PE = nn.Embedding(opt['max_len'], opt['dim_hidden'])

        if opt.get('TAP_ln', False):
            self.LN = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def forward(self, word_embeddings):
        if hasattr(self, 'PE'):
            seq_length = word_embeddings.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device)
            position_ids = position_ids.unsqueeze(0).repeat(word_embeddings.size(0), 1)
            position_embeddings = self.PE(position_ids)
            word_embeddings = word_embeddings + position_embeddings
        
        if hasattr(self, 'LN'):
            word_embeddings = self.LN(word_embeddings)
        
        word_embeddings = self.dropout(word_embeddings)
        return word_embeddings
