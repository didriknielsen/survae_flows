import os
import torch
from torchvision.datasets import FashionMNIST
from survae.data import DATA_PATH


class UnsupervisedFashionMNIST(FashionMNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(UnsupervisedFashionMNIST, self).__init__(root,
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
        return super(UnsupervisedFashionMNIST, self).__getitem__(index)[0]

    @property
    def raw_folder(self):
        # Replace self.__class__.__name__ by 'FashionMNIST'
        return os.path.join(self.root, 'FashionMNIST', 'raw')

    @property
    def processed_folder(self):
        # Replace self.__class__.__name__ by 'FashionMNIST'
        return os.path.join(self.root, 'FashionMNIST', 'processed')
