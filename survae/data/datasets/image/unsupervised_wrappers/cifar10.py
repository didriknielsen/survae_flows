import torch
from torchvision.datasets import CIFAR10
from survae.data import DATA_PATH


class UnsupervisedCIFAR10(CIFAR10):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(UnsupervisedCIFAR10, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(UnsupervisedCIFAR10, self).__getitem__(index)[0]
