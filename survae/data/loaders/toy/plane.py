import torch
from survae.data import TrainTestLoader
from survae.data.datasets.toy import GaussianDataset, CrescentDataset, CrescentCubedDataset, SineWaveDataset, AbsDataset, SignDataset, FourCirclesDataset, DiamondDataset, TwoSpiralsDataset, TwoMoonsDataset, TestGridDataset, CheckerboardDataset, FaceDataset


class Gaussian(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = GaussianDataset(num_points=train_samples)
        self.test = GaussianDataset(num_points=test_samples)


class Crescent(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = CrescentDataset(num_points=train_samples)
        self.test = CrescentDataset(num_points=test_samples)


class CrescentCubed(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = CrescentCubedDataset(num_points=train_samples)
        self.test = CrescentCubedDataset(num_points=test_samples)


class SineWave(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = SineWaveDataset(num_points=train_samples)
        self.test = SineWaveDataset(num_points=test_samples)


class Abs(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = AbsDataset(num_points=train_samples)
        self.test = AbsDataset(num_points=test_samples)


class Sign(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = SignDataset(num_points=train_samples)
        self.test = SignDataset(num_points=test_samples)


class FourCircles(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = FourCirclesDataset(num_points=train_samples)
        self.test = FourCirclesDataset(num_points=test_samples)


class Diamond(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100, width=20, bound=2.5, std=0.04):

        self.train = DiamondDataset(num_points=train_samples, width=width, bound=bound, std=std)
        self.test = DiamondDataset(num_points=test_samples, width=width, bound=bound, std=std)


class TwoSpirals(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = TwoSpiralsDataset(num_points=train_samples)
        self.test = TwoSpiralsDataset(num_points=test_samples)


class TwoMoons(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = TwoMoonsDataset(num_points=train_samples)
        self.test = TwoMoonsDataset(num_points=test_samples)


class Checkerboard(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100):

        self.train = CheckerboardDataset(num_points=train_samples)
        self.test = CheckerboardDataset(num_points=test_samples)


class Face(TrainTestLoader):

    def __init__(self, train_samples=100, test_samples=100, name='einstein', resize=[512,512]):

        self.train = FaceDataset(num_points=train_samples, name=name, resize=resize)
        self.test = FaceDataset(num_points=test_samples, name=name, resize=resize)
