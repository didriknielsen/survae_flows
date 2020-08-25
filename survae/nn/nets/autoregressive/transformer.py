import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.nn.layers import LambdaLayer
from survae.nn.layers.encoding import PositionalEncodingImage
from survae.nn.layers.autoregressive import AutoregressiveShift, Image2Seq, Seq2Image
from survae.nn.blocks.autoregressive import DecoderOnlyTransformer


class DecoderOnlyTransformer2d(nn.Module):
    '''An implementation of Decoder-only Transformers.'''

    def __init__(self, image_shape, output_dim, num_bits,
                 autoregressive_order='cwh', d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", kdim=None, vdim=None,
                 attn_bias=True, output_bias=True,
                 checkpoint_blocks=False,
                 in_lambda=lambda x: x,
                 out_lambda=lambda x: x):
        super(DecoderOnlyTransformer2d, self).__init__()
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

        self.transformer = DecoderOnlyTransformer(d_model=d_model,
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
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.encode(x)
        x = self.im2seq(x)
        x = self.ar_shift(x)
        x = self.transformer(x)
        x = self.out_linear(x)
        x = self.seq2im(x)
        return self.out_lambda(x)
