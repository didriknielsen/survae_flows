from survae.data.datasets.image import FixedBinaryMNISTDataset
from torch.utils.data import ConcatDataset
from survae.data import TrainValidTestLoader, DATA_PATH


class FixedBinarizedMNIST(TrainValidTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    with a fixed binarization as used in (Larochelle & Murray, 2011):
    http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf

    See Footnote 2 on page 6 and Appendix D of (Burda et al., 2015):
    https://arxiv.org/pdf/1509.00519.pdf
    for a remark on the different versions of MNIST.
    '''

    def __init__(self, root=DATA_PATH, download=True):

        self.train = FixedBinaryMNISTDataset(root, split='train', download=download)
        self.valid = FixedBinaryMNISTDataset(root, split='valid')
        self.test = FixedBinaryMNISTDataset(root, split='test')
