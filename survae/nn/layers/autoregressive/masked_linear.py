import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.utils import repeat_rows


# Adapted from https://github.com/bayesiains/nsf/blob/master/nde/transforms/made.py
# and https://github.com/karpathy/pytorch-made/blob/master/made.py

class MaskedLinear(nn.Linear):
    """
    A linear module with a masked weight matrix.

    Args:
        in_degrees: torch.LongTensor, length matching number of input features.
        out_features: int, number of output features.
        data_features: int, number of features in the data.
        random_mask: bool, if True, a random connection mask will be sampled.
        random_seed: int, seed used for sampling random order/mask.
        is_output: bool, whether the layer is the final layer.
        data_degrees: torch.LongTensor, length matching number of data features (needed if is_output=True).
        bias: bool, if True a bias is included.
    """

    def __init__(self,
                 in_degrees,
                 out_features,
                 data_features,
                 random_mask=False,
                 random_seed=None,
                 is_output=False,
                 data_degrees=None,
                 bias=True):
        if is_output:
            assert data_degrees is not None
            assert len(data_degrees) == data_features
        super(MaskedLinear, self).__init__(in_features=len(in_degrees),
                                           out_features=out_features,
                                           bias=bias)
        self.out_features = out_features
        self.data_features = data_features
        self.is_output = is_output
        mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
                                                      data_degrees=data_degrees,
                                                      random_mask=random_mask,
                                                      random_seed=random_seed)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', out_degrees)

    @staticmethod
    def get_data_degrees(in_features, random_order=False, random_seed=None):
        if random_order:
            rng = np.random.RandomState(random_seed)
            return torch.from_numpy(rng.permutation(in_features) + 1)
        else:
            return torch.arange(1, in_features + 1)

    def get_mask_and_degrees(self,
                             in_degrees,
                             data_degrees,
                             random_mask,
                             random_seed):
        if self.is_output:
            out_degrees = repeat_rows(data_degrees, self.out_features // self.data_features)
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, self.data_features - 1)
                rng = np.random.RandomState(random_seed)
                out_degrees = torch.from_numpy(rng.randint(min_in_degree,
                                                           self.data_features,
                                                           size=[self.out_features]))
            else:
                max_ = max(1, self.data_features - 1)
                min_ = min(1, self.data_features - 1)
                out_degrees = torch.arange(self.out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def update_mask_and_degrees(self,
                                in_degrees,
                                data_degrees,
                                random_mask,
                                random_seed):
        mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
                                                      data_degrees=data_degrees,
                                                      random_mask=random_mask,
                                                      random_seed=random_seed)
        self.mask.data.copy_(mask)
        self.degrees.data.copy_(out_degrees)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)
