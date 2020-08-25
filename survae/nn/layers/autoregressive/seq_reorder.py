import torch
import torch.nn as nn


class Image2Seq(nn.Module):

    def __init__(self, autoregressive_order, image_shape):
        assert autoregressive_order in {'cwh','whc','zigzag_cs'}
        super(Image2Seq, self).__init__()
        self.autoregressive_order = autoregressive_order
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        if autoregressive_order == 'zigzag_cs':
            self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

    def forward(self, x):
        b, dim = x.shape[0], x.shape[-1]
        l = x.shape[1:-1].numel()
        if self.autoregressive_order == 'whc':
            # x.shape: (b,c,h,w,dim)
            x = x.permute([1,2,3,0,4])
            # x.shape: (c,h,w,b,dim)
            x = x.reshape(l, b, dim)
            # x.shape: (l,b,dim)
        elif self.autoregressive_order == 'cwh':
            # x.shape: (b,c,h,w,dim)
            x = x.permute([2,3,1,0,4])
            # x.shape: (h,w,c,b,dim)
            x = x.reshape(l, b, dim)
            # x.shape: (l,b,dim)
        elif self.autoregressive_order == 'zigzag_cs':
            # x.shape: (b,c,h,w,dim)
            x = x[:, self.idx0, self.idx1, self.idx2, :]
            # x.shape: (b,l,dim)
            x = x.permute([1,0,2])
            # x.shape: (l,b,dim)
        return x


class Seq2Image(nn.Module):

    def __init__(self, autoregressive_order, image_shape):
        assert autoregressive_order in {'cwh','whc','zigzag_cs'}
        super(Seq2Image, self).__init__()
        self.autoregressive_order = autoregressive_order
        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        if autoregressive_order == 'zigzag_cs':
            self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

    def forward(self, x):
        b, dim = x.shape[1], x.shape[2]
        if self.autoregressive_order == 'whc':
            # x.shape: (l,b,dim)
            x = x.reshape(self.channels, self.height, self.width, b, dim)
            # x.shape: (c,h,w,b,dim)
            x = x.permute([3,0,1,2,4])
            # x.shape: (b,c,h,w,dim)
        elif self.autoregressive_order == 'cwh':
            # x.shape: (l,b,dim)
            x = x.reshape(self.height, self.width, self.channels, b, dim)
            # x.shape: (h,w,c,b,dim)
            x = x.permute([3,2,0,1,4])
            # x.shape: (b,c,h,w,dim)
        elif self.autoregressive_order == 'zigzag_cs':
            # x.shape: (l,b,dim)
            x = x.permute([1,0,2])
            # x.shape: (b,l,dim)
            y = torch.empty((x.shape[0], self.channels, self.height, self.width, x.shape[-1]), dtype=x.dtype, device=x.device)
            y[:, self.idx0, self.idx1, self.idx2, :] = x
            x = y
            # x.shape: (b,c,h,w,dim)
        return x


# Adapted from https://www.geeksforgeeks.org/print-matrix-zag-zag-fashion/
def _prep_zigzag_cs(channels, height, width):

    diagonals=[[] for i in range(height+width-1)]

    for i in range(height):
        for j in range(width):
            sum=i+j
            if(sum%2 ==0):

                #add at beginning
                diagonals[sum].insert(0,(i,j))
            else:

                #add at end of the list
                diagonals[sum].append((i,j))

    idx_list = []
    # print the solution as it as
    for d in diagonals:
        for idx in d:
            for c in range(channels):
                idx_list.append((c,)+idx)

    idx0, idx1, idx2 = zip(*idx_list)
    return idx0, idx1, idx2
