import torch
from torchvision.datasets import SVHN
from survae.data import DATA_PATH


class UnsupervisedSVHN(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False):
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
