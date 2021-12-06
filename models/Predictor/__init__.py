import os
import torch.nn as nn
from typing import Any, Dict, Optional   

from .pred_length import *
from .pred_attribute import *
from .base import Predictor


def add_predictor_specific_args(parser: object) -> object:
    for class_name in globals():
        if 'Predictor_' in class_name and hasattr(globals()[class_name], 'add_specific_args'):
            parser = globals()[class_name].add_specific_args(parser)
    
    return parser


def check_predictor_args(parser: object) -> None:
    for class_name in globals():
        if 'Predictor_' in class_name and hasattr(globals()[class_name], 'check_args'):
            globals()[class_name].check_args(parser)


def get_predictor(opt: Dict[str, Any]) -> Optional[nn.Module]:
    skip_crits = ['lang']

    nets = []
    for crit in opt['crits']:
        if crit in skip_crits:
            continue
        
        class_name = 'Predictor_{}'.format(crit)
            
        if class_name not in globals():
            path = os.path.join(os.path.dirname(__file__), class_name+'.py')
            raise ValueError('We can not find the class `{}` in {}'.format(class_name, path))
    
        nets.append(globals()[class_name](opt))

    if not len(nets):
        return None
    
    return Predictor(nets)
