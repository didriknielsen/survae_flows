import torch
import torch.nn as nn
from survae.nn.layers import GELU, Swish, ConcatReLU, ConcatELU

act_strs = {'elu', 'relu', 'gelu', 'swish'}
concat_act_strs = {'concat_elu', 'concat_relu'}


def act_module(act_str, allow_concat=False):
    if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
    else:            assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)
    if act_str == 'relu': return nn.ReLU()
    elif act_str == 'elu': return nn.ELU()
    elif act_str == 'gelu': return GELU()
    elif act_str == 'swish': return Swish()
    elif act_str == 'concat_relu': return ConcatReLU()
    elif act_str == 'concat_elu': return ConcatELU()


def act_factor(act_str, allow_concat=False):
    if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
    else:            assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)
    if act_str == 'relu': return 1
    elif act_str == 'elu': return 1
    elif act_str == 'gelu': return 1
    elif act_str == 'swish': return 1
    elif act_str == 'concat_relu': return 2
    elif act_str == 'concat_elu': return 2
