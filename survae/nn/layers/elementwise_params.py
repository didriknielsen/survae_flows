import torch
import torch.nn as nn


class ElementwiseParams(nn.Module):
    '''
    Move elementwise parameters to last dimension.
    Ex.: For an input of shape (B,D) with P elementwise parameters,
    the input takes shape (B,P*D) while the output takes shape (B,D,P).

    Args:
        num_params: int, number of elementwise parameters P.
        mode: str, mode of channels (see below), one of {'interleaved', 'sequential'} (default='interleaved').

    Mode:
        Ex.: For D=3 and P=2, the input is assumed to take the form along dimension 1:
        - interleaved: [1 2 3 1 2 3]
        - sequential: [1 1 2 2 3 3]
        while the output takes the form [1 2 3].
    '''

    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 2, 'Expected input of shape (B,D)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            dims = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * dims)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, dims))
                # x.shape = (bs, num_params, dims)
                x = x.permute([0,2,1])
                # x.shape = (bs, dims, num_params)
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (dims, self.num_params))
                # x.shape = (bs, dims, num_params)
        return x


class ElementwiseParams1d(nn.Module):
    '''
    Move elementwise parameters to last dimension.
    Ex.: For image of shape (B,D,L) with P elementwise parameters,
    the input takes shape (B,P*D,L) while the output takes shape (B,D,L,P).

    Args:
        num_params: int, number of elementwise parameters P.
        mode: str, mode of channels (see below), one of {'interleaved', 'sequential'} (default='interleaved').
    '''

    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams1d, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 3, 'Expected input of shape (B,D,L)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            dims = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * dims, length)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, dims) + x.shape[2:])
                # x.shape = (bs, num_params, dims, length)
                x = x.permute([0,2,3,1])
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (dims, self.num_params) + x.shape[2:])
                # x.shape = (bs, dims, num_params, length)
                x = x.permute([0,1,3,2])
            # x.shape = (bs, dims, length, num_params)
        return x


class ElementwiseParams2d(nn.Module):
    '''
    Move elementwise parameters to last dimension.
    Ex.: For image of shape (B,C,H,W) with P elementwise parameters,
    the input takes shape (B,P*C,H,W) while the output takes shape (B,C,H,W,P).

    Args:
        num_params: int, number of elementwise parameters P.
        mode: str, mode of channels (see below), one of {'interleaved', 'sequential'} (default='interleaved').

    Mode:
        Ex.: For C=3 and P=2, the input is assumed to take the form along the channel dimension:
        - interleaved: [R G B R G B]
        - sequential: [R R G G B B]
        while the output takes the form [R G B].
    '''

    def __init__(self, num_params, mode='interleaved'):
        super(ElementwiseParams2d, self).__init__()
        assert mode in {'interleaved', 'sequential'}
        self.num_params = num_params
        self.mode = mode

    def forward(self, x):
        assert x.dim() == 4, 'Expected input of shape (B,C,H,W)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            channels = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * channels , height, width)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, channels) + x.shape[2:])
                # x.shape = (bs, num_params, channels, height, width)
                x = x.permute([0,2,3,4,1])
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (channels, self.num_params) + x.shape[2:])
                # x.shape = (bs, channels, num_params, height, width)
                x = x.permute([0,1,3,4,2])
            # x.shape = (bs, channels, height, width, num_params)
        return x
