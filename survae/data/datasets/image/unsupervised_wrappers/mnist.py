import os
import torch
from torchvision.datasets import MNIST
from survae.data import DATA_PATH


class UnsupervisedMNIST(MNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(UnsupervisedMNIST, self).__init__(root,
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
        return super(UnsupervisedMNIST, self).__getitem__(index)[0]

    @property
    def raw_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'processed')
