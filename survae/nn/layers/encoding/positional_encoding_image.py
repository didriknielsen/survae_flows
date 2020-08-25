import math
import torch
import torch.nn as nn


class PositionalEncodingImage(nn.Module):
    '''
    Learning positional embeddings for images.
    Embeddings for channel, height and width are added to form the full positional embedding.
    These encodings correspond to the ones from Sparse Transformers (https://arxiv.org/abs/1904.10509).

    Args:
        image_shape: Iterable, the shape of the image.
        embedding_dim: int, the size of each embedding vector.
    '''

    def __init__(self, image_shape, embedding_dim):
        super(PositionalEncodingImage, self).__init__()
        assert len(image_shape) == 3, 'image_shape should have length 3: (C,H,W)'
        self.image_shape = image_shape
        self.embedding_dim = embedding_dim

        c, h, w = image_shape
        self.encode_c = nn.Parameter(torch.Tensor(1, c, 1, 1, embedding_dim))
        self.encode_h = nn.Parameter(torch.Tensor(1, 1, h, 1, embedding_dim))
        self.encode_w = nn.Parameter(torch.Tensor(1, 1, 1, w, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize position embedding layers to N(0, 0.125/sqrt(3*d))
        # as described in paragraph 3 of Section "6. Training":
        nn.init.normal_(self.encode_c, std=0.125/math.sqrt(3*self.embedding_dim))
        nn.init.normal_(self.encode_h, std=0.125/math.sqrt(3*self.embedding_dim))
        nn.init.normal_(self.encode_w, std=0.125/math.sqrt(3*self.embedding_dim))

    def forward(self, x):
        return x + self.encode_c + self.encode_h + self.encode_w
