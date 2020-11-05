import torch
from torchvision.datasets import SVHN
from survae.data import DATA_PATH


class UnsupervisedSVHN(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False):
        """
        Initialize the root directory.

        Args:
            self: (todo): write your description
            root: (str): write your description
            DATA_PATH: (str): write your description
            split: (int): write your description
            transform: (str): write your description
            download: (todo): write your description
        """
        super(UnsupervisedSVHN, self).__init__(root,
                                               split=split,
                                               transform=transform,
                                               download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(UnsupervisedSVHN, self).__getitem__(index)[0]
