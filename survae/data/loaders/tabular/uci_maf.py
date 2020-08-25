from survae.data.datasets.tabular import BSDS300Dataset, GasDataset, HEPMASSDataset, MiniBooNEDataset, PowerDataset
from survae.data import TrainValidTestLoader, DATA_PATH


class BSDS300(TrainValidTestLoader):

    def __init__(self, root=DATA_PATH, validation=False, download=True):
        self.root = root
        self.train = BSDS300Dataset(root, split='train', download=download)
        self.valid = BSDS300Dataset(root, split='validation', download=download)
        self.test = BSDS300Dataset(root, split='test', download=download)


class Gas(TrainValidTestLoader):

    def __init__(self, root=DATA_PATH, validation=False, download=True):
        self.root = root
        self.train = GasDataset(root, split='train', download=download)
        self.valid = GasDataset(root, split='validation', download=download)
        self.test = GasDataset(root, split='test', download=download)


class HEPMASS(TrainValidTestLoader):

    def __init__(self, root=DATA_PATH, validation=False, download=True):
        self.root = root
        self.train = HEPMASSDataset(root, split='train', download=download)
        self.valid = HEPMASSDataset(root, split='validation', download=download)
        self.test = HEPMASSDataset(root, split='test', download=download)


class MiniBooNE(TrainValidTestLoader):

    def __init__(self, root=DATA_PATH, validation=False, download=True):
        self.root = root
        self.train = MiniBooNEDataset(root, split='train', download=download)
        self.valid = MiniBooNEDataset(root, split='validation', download=download)
        self.test = MiniBooNEDataset(root, split='test', download=download)


class Power(TrainValidTestLoader):

    def __init__(self, root=DATA_PATH, validation=False, download=True):
        self.root = root
        self.train = PowerDataset(root, split='train', download=download)
        self.valid = PowerDataset(root, split='validation', download=download)
        self.test = PowerDataset(root, split='test', download=download)
