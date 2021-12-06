import torch
import torch.nn as nn
from torch.nn.modules import module  

__all__ = (
    'NaiveHead', 
    'MLPHead',
)


def get_cls_head(opt: dict) -> nn.Module:
    class_name = opt['cls_head']
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


class HeadBase(nn.Module):
    def get_word_embeddings(self):
        return self.tgt_word_prj.weight

    def set_word_embeddings(self, data, add_bias=False):
        assert isinstance(data, nn.Parameter)
        self.tgt_word_prj.weight = data
        if add_bias and self.tgt_word_prj.bias is None:
            vocab_size = self.tgt_word_prj.weight.size(0)
            self.tgt_word_prj.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)


class NaiveHead(HeadBase):
    def __init__(self, opt):
        super().__init__()
        self.tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=False)

    def forward(self, hidden_states):
        return self.tgt_word_prj(hidden_states)


class MLPHead(HeadBase):
    '''`Hierarchical LSTM with Adjusted Temporal Attention for Video Captioning` (IJCAI 2017)
        https://www.ijcai.org/proceedings/2017/381
    '''
    def __init__(self, opt):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(opt['dim_hidden'] * 2, opt['dim_hidden'], bias=True),
            nn.Tanh(),
            nn.Dropout(opt['hidden_dropout_prob'])
        )
        self.tgt_word_prj = nn.Linear(opt["dim_hidden"], opt["vocab_size"], bias=True)

    def forward(self, hidden_states):
        return self.tgt_word_prj(self.dense(hidden_states))
