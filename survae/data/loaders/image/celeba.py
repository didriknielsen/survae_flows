from survae.data.datasets.image import CelebADataset
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainValidTestLoader, DATA_PATH


class CelebA(TrainValidTestLoader):
    '''
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    '''

    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = CelebADataset(root, split='train', transform=Compose(trans_train))
        self.valid = CelebADataset(root, split='valid', transform=Compose(trans_test))
        self.test = CelebADataset(root, split='test', transform=Compose(trans_test))
