import os
from torch.utils.data import DataLoader
from datasets import SpatialMNISTDataset
from survae.data import DATA_PATH

dataset_choices = {'spatial_mnist'}


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == 'spatial_mnist':
        dataset = DataContainer(SpatialMNISTDataset(os.path.join(DATA_PATH, 'spatial_mnist'), split='train'),
                                SpatialMNISTDataset(os.path.join(DATA_PATH, 'spatial_mnist'), split='valid'),
                                SpatialMNISTDataset(os.path.join(DATA_PATH, 'spatial_mnist'), split='test'))

    # Data Loader
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset.valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class DataContainer():
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
