import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.nn.layers import LambdaLayer
from survae.nn.layers.encoding import PositionalEncodingImage
from survae.nn.layers.autoregressive import AutoregressiveShift, Image2Seq, Seq2Image
from survae.nn.blocks.autoregressive import DenseTransformer


class DenseTransformer2d(nn.Module):
    '''An implementation of Sparse Transformers (https://arxiv.org/abs/1904.10509) using dense self-attention.'''

    def __init__(self, image_shape, output_dim, num_bits,
                 autoregressive_order='cwh', d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", kdim=None, vdim=None,
                 attn_bias=True, output_bias=True,
                 checkpoint_blocks=False,
                 in_lambda=lambda x: x,
                 out_lambda=lambda x: x):
        super(DenseTransformer2d, self).__init__()
        self.image_shape = torch.Size(image_shape)
        self.autoregressive_order = autoregressive_order
        self.d_model = d_model
        self.num_layers = num_layers

        # Encoding layers
        self.encode = nn.Sequential(LambdaLayer(in_lambda),
                                    nn.Embedding(2**num_bits, d_model),
                                    PositionalEncodingImage(image_shape=image_shape, embedding_dim=d_model))

        self.im2seq = Image2Seq(autoregressive_order, image_shape)
        self.seq2im = Seq2Image(autoregressive_order, image_shape)
        self.ar_shift = AutoregressiveShift(d_model)

        self.transformer = DenseTransformer(d_model=d_model,
                                            nhead=nhead,
                                            num_layers=num_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            activation=activation,
                                            kdim=kdim,
                                            vdim=vdim,
                                            attn_bias=attn_bias,
                                            checkpoint_blocks=checkpoint_blocks)

        self.out_linear = nn.Linear(d_model, output_dim, bias=output_bias)
        self.out_lambda = LambdaLayer(out_lambda)

        self._reset_parameters()

    def _reset_parameters(self):
        # The initialization is done as described in:
        # - Paragraph 3 of Section "6. Training".
        # - Final paragraph of Section "5.2. Scaling to hundreds of layers".
        # Note that the initializations described has already been performed for
        # parameters in DenseTransformer and PositionalEncodingImage.
        # What remains to initialize is thus:
        # 1) The token embedding
        # 2) The output layer

        # Initialize output weight matrix to 0:
        nn.init.zeros_(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)

        # Initialize token embedding layers to N(0, 0.125/sqrt(d)):
        nn.init.normal_(self.encode._modules['1'].weight, std=0.125/math.sqrt(self.d_model))

    def forward(self, x):
        x = self.encode(x)
        x = self.im2seq(x)
        x = self.ar_shift(x)
        x = self.transformer(x)
        x = self.out_linear(x)
        x = self.seq2im(x)
        return self.out_lambda(x)
