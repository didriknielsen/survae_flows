import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split


class PresplitLoader():

    @property
    def num_splits(self):
        return len(self.splits)

    def get_data_loader(self, split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
        return DataLoader(getattr(self, split), batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
        data_loaders = [self.get_data_loader(split=split,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             pin_memory=pin_memory,
                                             num_workers=num_workers) for split in self.splits]
        return data_loaders


class TrainTestLoader(PresplitLoader):
    splits = ['train', 'test']


class TrainValidTestLoader(PresplitLoader):
    splits = ['train', 'valid', 'test']

    @property
    def train_valid(self):
        return ConcatDataset([self.train, self.valid])


class RandomSplitLoader(TrainTestLoader):

    def random_split(self, dataset, test_fraction=0.5, split_seed=0):
        train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_fraction, random_state=split_seed)
        train = Subset(dataset, train_idx)
        test = Subset(dataset, test_idx)
        return train, test
