from survae.data.datasets.image import ImageNet64Dataset
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class ImageNet64(TrainTestLoader):
    '''
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 64x64, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = ImageNet64Dataset(root, train=True, transform=Compose(trans_train), download=download)
        self.test = ImageNet64Dataset(root, train=False, transform=Compose(trans_test))
