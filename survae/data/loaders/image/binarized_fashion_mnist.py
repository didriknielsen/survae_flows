from survae.data.datasets.image import UnsupervisedFashionMNIST
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Flatten, DynamicBinarize
from survae.data import TrainTestLoader, DATA_PATH


class DynamicallyBinarizedFashionMNIST(TrainTestLoader):
    '''
    The Fasion-MNIST dataset of (Xiao et al., 2017):
    https://arxiv.org/abs/1708.07747
    with a dynamic binarization.
    '''

    def __init__(self, root=DATA_PATH, download=True, flatten=False):

        self.root = root

        # Define transformations
        trans = [ToTensor(), DynamicBinarize()]
        if flatten: trans.append(Flatten())

        # Load data
        self.train = UnsupervisedFashionMNIST(root, train=True, transform=Compose(trans), download=download)
        self.test = UnsupervisedFashionMNIST(root, train=False, transform=Compose(trans))
