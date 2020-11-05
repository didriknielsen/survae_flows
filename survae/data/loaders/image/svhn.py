import os
from survae.data.datasets.image import UnsupervisedSVHN
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class SVHN(TrainTestLoader):
    '''
    The SVHN dataset of (Netzer et al., 2011):
    https://research.google/pubs/pub37648/
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):
        """
        Initialize a tensor

        Args:
            self: (todo): write your description
            root: (str): write your description
            DATA_PATH: (str): write your description
            download: (todo): write your description
            num_bits: (int): write your description
            pil_transforms: (todo): write your description
        """

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        sub_root = os.path.join(root, 'SVHN')
        self.train = UnsupervisedSVHN(sub_root, split='train', transform=Compose(trans_train), download=download)
        self.test = UnsupervisedSVHN(sub_root, split='test', transform=Compose(trans_test), download=download)
